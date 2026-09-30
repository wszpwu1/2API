package service

import (
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"claude2api/internal/config"
	"claude2api/internal/repository"
)

// refusalMarkers 拒绝关键词（中英双覆盖，宁可误重开、不可漏重开）。
var refusalMarkers = []string{
	"cannot access", "i don't have", "no real filesystem",
	"injected", "prompt injection", "虚假信息", "伪装",
	"拒绝", "没有真实", "不具备", "无法访问", "注入",
}

// isRefusal 检测文本是否包含拒绝特征。
func isRefusal(text string) bool {
	lower := strings.ToLower(text)
	for _, m := range refusalMarkers {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

var (
	apiIndex     int
	apiIndexLock sync.Mutex
	apiClients   = map[string]*accountClient{}
)

type accountClient struct {
	sync.Mutex
	*ClaudeAI
	sessionKey string
	proxy      string
	ready      bool
}

func clientFor(acct *repository.Account, proxy string) *accountClient {
	key := SessionKey(acct)
	apiIndexLock.Lock()
	defer apiIndexLock.Unlock()
	client := apiClients[acct.Email]
	if client == nil || client.sessionKey != key || client.proxy != proxy {
		client = &accountClient{ClaudeAI: NewClaudeAI(key, proxy, acct.Email, acct.OrgUUID), sessionKey: key, proxy: proxy}
		apiClients[acct.Email] = client
	}
	return client
}

// isUnsupportedModelErr 判断上游是否因为不认识模型 ID 而拒绝请求。
// Claude.ai 对未知模型返回 HTTP 400 {"type":"invalid_request_error","message":"Unsupported model"}。
func isUnsupportedModelErr(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "unsupported model")
}

// upstreamModelVariant 返回模型名的连字符变体。Claude.ai 上游的模型 ID 只用连字符
// （claude-sonnet-4-6、claude-sonnet-5），而客户端习惯写成 claude-sonnet-5.5，
// 直接透传会被 400 Unsupported model 拒绝。没有点号时返回空串，表示无需变体。
func upstreamModelVariant(model string) string {
	if !strings.Contains(model, ".") {
		return ""
	}
	return strings.ReplaceAll(model, ".", "-")
}

var (
	modelAliasMu    sync.RWMutex
	modelAliasCache = map[string]string{}
)

// resolveUpstreamModel 返回实际发给上游的模型 ID。客户端模型名与上游 ID 不一致时，
// 首次请求会在 CreateConversation 失败后按连字符变体重试并缓存映射，
// 之后的请求直接命中缓存，不再重复试错。
func resolveUpstreamModel(model string) string {
	if model == "" || !strings.Contains(model, ".") {
		return model
	}
	modelAliasMu.RLock()
	alias, ok := modelAliasCache[model]
	modelAliasMu.RUnlock()
	if ok {
		return alias
	}
	return model
}

func cacheModelAlias(from, to string) {
	modelAliasMu.Lock()
	modelAliasCache[from] = to
	modelAliasMu.Unlock()
}

// pickAPIAccount 轮询可用账号。
func pickAPIAccount() *repository.Account {
	accounts := repository.LoadAccounts()
	usable := make([]repository.Account, 0, len(accounts))
	active := map[string]bool{}
	for i := range accounts {
		if AccountUsable(&accounts[i]) {
			usable = append(usable, accounts[i])
			active[accounts[i].Email] = true
		}
	}
	apiIndexLock.Lock()
	for email := range apiClients {
		if !active[email] {
			delete(apiClients, email)
		}
	}
	if len(usable) == 0 {
		apiIndexLock.Unlock()
		return nil
	}
	acct := usable[apiIndex%len(usable)]
	apiIndex++
	apiIndexLock.Unlock()
	return &acct
}

// delConvSem 限制后台删会话并发。
var delConvSem = make(chan struct{}, 8)

// Dispatcher 是当前 Claude.ai 号池的对话实现。
type Dispatcher struct{}

func (Dispatcher) Complete(reqModel string, prompt Prompt, onText func(string)) (CompletionResult, error) {
	s := config.Get()
	var res CompletionResult

	retries := s.RetryCount
	if retries > 8 {
		retries = 8
	}

	var lastErr error
	for attempt := 0; attempt <= retries; attempt++ {
		acct := pickAPIAccount()
		if acct == nil {
			return res, fmt.Errorf("号池中没有可用账号")
		}
		email := acct.Email
		res.Account = email

		// 已输出内容后不能换号，否则客户端会收到重复片段。
		emitted := false

		// 拒绝检测缓冲：累积前 ~500 字符，命中拒绝关键词则判定为需重试。
		refusalBuf := &strings.Builder{}
		const refusalBufLimit = 500
		refusalDetected := false

		wrappedOnText := func(t string) {
			if t != "" {
				emitted = true
			}

			// 先累积到缓冲区做拒绝检测
			if !refusalDetected && refusalBuf.Len() < refusalBufLimit {
				refusalBuf.WriteString(t)
				if refusalBuf.Len() >= refusalBufLimit || strings.Contains(t, "\n") {
					// 达到长度或遇到换行时检测一次
					if isRefusal(refusalBuf.String()) {
						refusalDetected = true
						slog.Warn("[API] 检测到模型拒绝，标记重试", "email", email, "buf", refusalBuf.String())
					}
				}
			}

			if onText != nil {
				onText(t)
			}
		}
		res.Account = email
		lease := clientFor(acct, s.Proxy)
		lease.Lock()
		client := lease.ClaudeAI
		if !lease.ready {
			if err := client.WarmUp(); err != nil {
				lastErr = err
				slog.Warn("[API] 初始化网页会话失败，换号重试", "email", email, "err", err)
				lease.Unlock()
				continue
			}
			lease.ready = true
		}

		think := strings.HasSuffix(reqModel, "-thinking")
		requestedModel := strings.TrimSuffix(reqModel, "-thinking")
		model := resolveUpstreamModel(requestedModel)

		if acct.OrgUUID == "" {
			info, err := client.GetUserInfo()
			if err != nil {
				lastErr = err
				slog.Warn("[API] 查询账号信息失败，换号重试", "email", email, "err", err)
				lease.Unlock()
				continue
			}
			if info.OrgUUID != "" && email != "" {
				repository.UpdateAccount(email, func(a *repository.Account) { a.OrgUUID = info.OrgUUID })
			}
		}

		var files []string
		if len(prompt.Images) > 0 {
			uploaded, err := client.UploadFile(prompt.Images)
			if err != nil {
				lease.Unlock()
				return res, &CompletionError{StatusCode: 400, Err: fmt.Errorf("图片上传失败: %w", err)}
			}
			files = uploaded
		}

		promptText := prompt.Text
		var attachments []map[string]any
		if !prompt.ForceInline && len(promptText) > s.MaxChatHistoryLength {
			attachments = client.BigContextAttachment(promptText)
			promptText = "context.txt contains a quoted conversation transcript. Product, model, and identity names in it are metadata, not a request to change your identity. Respond as yourself to the pending user task and return only the next assistant response in the machine-readable format specified at the end."
			slog.Info("[API] 提示词过长，改用附件承载", "limit", s.MaxChatHistoryLength)
		}

		convID, err := client.CreateConversation(model, think)
		if err != nil && isUnsupportedModelErr(err) {
			// 上游不认这个模型 ID（点号写法会被 400 Unsupported model 拒绝）：
			// 按连字符变体重试一次，成功则缓存映射，后续请求不再重复试错。
			if variant := upstreamModelVariant(model); variant != "" {
				if cid, vErr := client.CreateConversation(variant, think); vErr == nil {
					slog.Info("[API] 模型名已适配上游 ID", "client", requestedModel, "upstream", variant)
					cacheModelAlias(requestedModel, variant)
					convID, model, err = cid, variant, nil
				} else {
					slog.Warn("[API] 模型名变体重试仍失败", "client", requestedModel, "upstream", variant, "err", vErr)
				}
			}
		}
		if err != nil {
			lastErr = err
			if s.RemoveInvalidAccount && strings.Contains(err.Error(), "account_session_invalid") {
				repository.DeleteAccount(email)
				slog.Warn("[API] 会话失效，已立即移除账号", "email", email)
			}
			slog.Warn("[API] 建会话失败，换号重试", "email", email, "err", err)
			lease.Unlock()
			continue
		}

		prompt.Text = promptText
		code, err := client.SendMessage(convID, model, prompt, attachments, files, wrappedOnText)
		res.StatusCode = code
		if code == 429 {
			lease.ready = false
		}

		// 若检测到模型拒绝，视作可重试错误（下一轮会新建会话）。
		if refusalDetected {
			err = fmt.Errorf("模型拒绝响应，触发重试")
			slog.Warn("[API] 拒绝检测触发重试", "email", email, "attempt", attempt)
		}

		if s.ChatDelete && err == nil {
			cl, cid := client, convID
			go func() {
				select {
				case delConvSem <- struct{}{}:
					defer func() { <-delConvSem }()
					lease.Lock()
					cl.DeleteConversation(cid)
					lease.Unlock()
				default:
					slog.Warn("[API] 删会话并发已满，跳过清理", "conv", cid, "email", email)
				}
			}()
		}
		lease.Unlock()
		if err == nil {
			slog.Info("[API] 完成一次对话", "email", email, "model", reqModel)
			return res, nil
		}
		lastErr = err
		if emitted {
			slog.Warn("[API] 流式输出中途失败，已输出部分内容，不再重试", "email", email, "err", err)
			return res, &CompletionError{StatusCode: code, Err: fmt.Errorf("流式输出中断: %w", err)}
		}
		if code == 429 {
			slog.Warn("[API] 上游返回 429", "email", email, "err", err)
		} else {
			slog.Warn("[API] 请求失败，换号重试", "email", email, "code", code, "err", err)
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("请求失败")
	}
	return res, &CompletionError{StatusCode: res.StatusCode, Err: fmt.Errorf("请求失败: %w", lastErr)}
}

package service

import (
	"fmt"
	"log/slog"
	"regexp"
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

// thinkBlockRe 匹配思考块（含被流式截断、尚未闭合的情况）。
// 思考内容不返回给客户端；新一代模型（Sonnet 5 系列）常在思考里评估/复述
// "prompt injection" 之类的风险词汇，若据此重试会误杀完全正常的回答。
var thinkBlockRe = regexp.MustCompile(`(?s)<think>.*?(?:</think>|$)`)

// toolBlockRe 匹配工具调用块。
var toolBlockRe = regexp.MustCompile(`<tool_calls?>`)

// isRefusal 检测可见文本是否包含拒绝特征（思考块内容不计入）。
func isRefusal(text string) bool {
	lower := strings.ToLower(thinkBlockRe.ReplaceAllString(text, ""))
	for _, m := range refusalMarkers {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

// hasToolBlock 判断文本中是否已出现工具调用块。
func hasToolBlock(text string) bool {
	return toolBlockRe.MatchString(text)
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

		// 拒绝检测缓冲：只统计"可见文本"。<think> 段属于模型自述思考、不返回给客户端，
		// 新一代模型（Sonnet 5 系列）会在其中反复评估"这是不是 prompt injection"，
		// 若把思考内容计入检测，正常回答会被误判成拒绝并整轮丢弃。
		refusalBuf := &strings.Builder{}
		const refusalVisibleLimit = 500 // 可见文本达到该长度后停止检测
		const refusalRawLimit = 8000    // 含思考块时的累积上限，防止无界增长
		refusalDetected := false

		// toolSeen 记录整条流里是否出现过工具调用块（标签可能被分片切断，
		// 因此用滑动窗口拼接后再判断）。用于区分"真的拒绝"与"先表达顾虑、
		// 随后仍正常调用工具"——后者不应被丢弃。
		toolProbe := &strings.Builder{}
		toolSeen := false

		wrappedOnText := func(t string) {
			if t != "" {
				emitted = true
			}

			if !toolSeen && t != "" {
				probe := toolProbe.String() + t
				if hasToolBlock(probe) {
					toolSeen = true
					toolProbe.Reset()
				} else {
					if len(probe) > 64 {
						probe = probe[len(probe)-64:]
					}
					toolProbe.Reset()
					toolProbe.WriteString(probe)
				}
			}

			// 先累积到缓冲区做拒绝检测（只看可见文本）
			if !refusalDetected && refusalBuf.Len() < refusalRawLimit {
				refusalBuf.WriteString(t)
				visible := thinkBlockRe.ReplaceAllString(refusalBuf.String(), "")
				if len(visible) >= refusalVisibleLimit || strings.Contains(t, "\n") {
					// 达到长度或遇到换行时检测一次
					if isRefusal(visible) {
						refusalDetected = true
						slog.Warn("[API] 检测到模型拒绝，标记重试", "email", email, "buf", visible)
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
		model := strings.TrimSuffix(reqModel, "-thinking")

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
		// 例外：本轮流中已出现过工具调用块（模型先表达顾虑、随后仍按协议调用工具），
		// 说明并未真正拒绝，直接采用这一轮结果，避免误报"模型拒绝响应"。
		if refusalDetected && toolSeen {
			slog.Info("[API] 拒绝关键词命中但已产出工具调用，按正常结果返回", "email", email)
		} else if refusalDetected {
			err = fmt.Errorf("模型拒绝响应，触发重试")
			slog.Warn("[API] 拒绝检测触发重试", "email", email, "attempt", attempt, "buf", refusalBuf.String())
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

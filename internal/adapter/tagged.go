package adapter

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// 标签工具协议（tagged tool protocol）：claude.ai 网页端只能输出纯文本，
// 因此让模型只输出一套强约束标签 DSL，服务端再解析为协议无关的中间语义，
// 最后由各协议 handler 输出成 OpenAI / Anthropic 的 tool_calls / tool_use。
//
// 模型输出契约：
//
//	<think>...</think>
//	<tool_calls>[{"name":"Read","arguments":{"path":"..."}}]</tool_calls>
//
// 或：
//
//	<think>...</think>
//	<final_answer>给用户的最终回复</final_answer>

const taggedToolPromptParallel = `Write the AI assistant's next response using only the following XML-like tags:

- <think>...</think>
- <tool_calls>[{"name":"ToolName","arguments":{...}}]</tool_calls>
- <final_answer>...</final_answer>

Rules:
- You may output one or more <think> blocks.
- You must then output exactly one terminal block: either <tool_calls> or <final_answer>.
- Do not output any text outside these tags.
- In <tool_calls>, the content must be a valid JSON array. Each item must be an object with keys "name" and "arguments".
- If you need only one tool, still use <tool_calls> with an array of length 1.
- In string values inside <tool_calls>, you must escape quotes, backslashes, and newlines exactly as JSON requires.
- After </tool_calls> or </final_answer>, stop immediately.
- Never generate Observation, tool results, or a second terminal block in the same response.
- Never output <observation>; the system will provide tool results in the next turn.
`

const taggedToolPromptSingle = `Write the AI assistant's next response using only the following XML-like tags:

- <think>...</think>
- <tool_call>{"name":"ToolName","arguments":{...}}</tool_call>
- <final_answer>...</final_answer>

Rules:
- You may output one or more <think> blocks.
- You must then output exactly one terminal block: either <tool_call> or <final_answer>.
- Do not output any text outside these tags.
- In <tool_call>, the content must be a valid JSON object with keys "name" and "arguments".
- In string values inside <tool_call>, you must escape quotes, backslashes, and newlines exactly as JSON requires.
- After </tool_call> or </final_answer>, stop immediately.
- Never generate Observation, tool results, or a second terminal block in the same response.
- Never output <observation>; the system will provide tool results in the next turn.
`

// taggedToolDirective 强化指令：claude.ai 网页端作为上游没有真正的 function
// calling，工具只是以纯文本形式给出。模型极易"礼貌地"用"我没有联网能力"之类
// 的话拒绝调用工具，从而被容错解析降级成普通回答。此段明确禁止此类拒绝，
// 并要求对需要实时/外部信息的问题强制使用工具，提升标签协议的实际触发率。
const taggedToolDirective = `
IMPORTANT — read carefully before responding:
- The tools listed below are REAL and available to you right now. You are NOT blocked from using them.
- If the request needs CURRENT, REAL-TIME, or EXTERNAL information (weather, news, prices, live status, or anything you cannot answer from the conversation alone), you MUST call the appropriate tool. Do not answer from memory when a tool exists.
- NEVER reply with "I don't have internet access", "I can't browse", "I lack tools", or any similar refusal. Those tools are provided for exactly this purpose.
- Only emit <final_answer> when you can fully answer using the conversation history alone. When in doubt, prefer calling a tool.
- You also have LOCAL FILESYSTEM and CODE-EXECUTION tools (Read, Edit, Write, Bash, Grep, Glob). When the user refers to local files, project code, or asks you to make a change, you MUST use these tools — do not say you cannot access files or cannot edit code.
- NEVER reply that you cannot access local files, cannot run commands, or cannot edit the user's code. Such tools are provided and you are expected to use them.
- When a file is referenced without an absolute path, use Glob/Grep to locate it; do not claim it cannot be found.
- If the user asks you to write or modify code, you MUST actually perform the change with Edit/Write/Bash — do not only describe it in text and refuse to apply it.
`

// taggedToolReminder 放在 prompt 末尾（紧邻生成位置）复述工具能力。
// 原因：claude.ai 上游没有真正的 function calling，工具协议只在 prompt 开头以纯文本声明；
// Claude Code 多轮长对话里，开头的 taggedToolDirective 会被越堆越长的历史与工具结果稀释，
// 模型会"忘记"自己具备调用能力、转而模仿它自己上一轮的纯文本 <final_answer> 而拒绝调用工具。
// 在生成前的最后位置重申一次，可显著提升实际触发率（等价于每轮自动提醒）。
const taggedToolReminder = `<tool_use_reminder>
You are inside an agentic tool loop right now. The tools listed above are REAL, live, and callable by you THIS turn.
- For ANY request needing current / real-time / external information or any action on files, the web, or the system (weather, news, prices, live status, reading/editing files, running commands, searching), you MUST emit a <tool_call> (or <tool_calls>) block now — do NOT answer from memory and do NOT claim you lack the ability.
- Do not imitate a previous plain-text answer: each turn you decide fresh whether a tool is needed.
- Emit <final_answer> ONLY when this turn already contains everything required to fully answer, with no tool needed.
When in doubt, prefer calling the tool.
- You have REAL local filesystem and code-execution tools (Read, Edit, Write, Bash, Grep, Glob). When the user asks about local files, project code, or wants a change applied, you MUST call them — never say "I cannot access local files" or "I cannot edit your code".
- If a file is referenced without an absolute path, use Glob/Grep to locate it; do not claim it cannot be found.
- When asked to write or modify code, actually apply the change with Edit/Write/Bash; do not only describe it in text.
</tool_use_reminder>`

// taggedToolParseFailedNotice 是工具块无法解析时给客户端的提示文本。
// 宁可显式告知，也不要把 <tool_calls>[…] 协议原文当正文透传（用户/终端会直接看到参数）。
const taggedToolParseFailedNotice = "\n[tool_call protocol error] 上游返回的工具调用格式无法解析，本轮工具调用已被忽略，请重试。"

// TaggedToolCall 是解析出的单次工具调用。
type TaggedToolCall struct {
	Name      string
	Arguments map[string]any
}

// TaggedOutput 是模型一轮响应的解析结果。
type TaggedOutput struct {
	Thinking    string
	ToolCalls   []TaggedToolCall
	FinalAnswer string
	HasFinal    bool
}

// IsToolCall 表示本轮是工具调用。
func (o TaggedOutput) IsToolCall() bool { return len(o.ToolCalls) > 0 }

// IsFinalAnswer 表示本轮是最终回答。
func (o TaggedOutput) IsFinalAnswer() bool { return o.HasFinal }

// FormatToolsForPrompt 把 OpenAI / Anthropic 风格的 tools 转成可读文本列表。
func FormatToolsForPrompt(tools []map[string]any) string {
	if len(tools) == 0 {
		return ""
	}
	lines := make([]string, 0, len(tools))
	for _, tool := range tools {
		fn := tool
		if t, _ := tool["type"].(string); t == "function" {
			if f, ok := tool["function"].(map[string]any); ok {
				fn = f
			}
		}
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		description, _ := fn["description"].(string)
		if description == "" {
			if s, ok := fn["summary"].(string); ok {
				description = s
			}
		}
		params := mapField(fn, "parameters")
		if params == nil {
			params = mapField(fn, "input_schema")
		}
		var argsDesc string
		if params != nil {
			props := mapField(params, "properties")
			required := map[string]bool{}
			if reqList, ok := params["required"].([]any); ok {
				for _, r := range reqList {
					if s, ok := r.(string); ok {
						required[s] = true
					}
				}
			}
			if props != nil {
				argParts := make([]string, 0, len(props))
				for key, v := range props {
					vm, ok := v.(map[string]any)
					if !ok {
						continue
					}
					typ, _ := vm["type"].(string)
					if typ == "" {
						typ = "any"
					}
					part := fmt.Sprintf("%s: %s", key, typ)
					if required[key] {
						part += " (required)"
					}
					argParts = append(argParts, part)
				}
				argsDesc = strings.Join(argParts, ", ")
			}
		}
		suffix := ""
		if len(description) > 200 {
			description = description[:200]
			suffix = "..."
		}
		lines = append(lines, fmt.Sprintf("- %s(%s): %s%s", name, argsDesc, description, suffix))
	}
	return strings.Join(lines, "\n")
}

// mapField 取一个可能是 map 的字段。
func mapField(m map[string]any, key string) map[string]any {
	if v, ok := m[key].(map[string]any); ok {
		return v
	}
	return nil
}

// FormatTaggedPrompt 构造标签协议的系统前缀。
func FormatTaggedPrompt(tools []map[string]any, allowParallel bool) string {
	toolsText := FormatToolsForPrompt(tools)
	base := taggedToolPromptParallel
	if !allowParallel {
		base = taggedToolPromptSingle
	}
	if toolsText != "" {
		return base + taggedToolDirective + "\n\n---\n\n## Available tools\n\n" + toolsText + "\n"
	}
	return base
}

// normalizeToolArguments 接受模型常见的 object、JSON 字符串、null 和缺省参数形式。
func normalizeToolArguments(value any) (map[string]any, error) {
	if value == nil {
		return map[string]any{}, nil
	}
	switch v := value.(type) {
	case map[string]any:
		return v, nil
	case string:
		var decoded any
		if err := json.Unmarshal([]byte(strings.TrimSpace(v)), &decoded); err != nil {
			return nil, fmt.Errorf("tool_call.arguments string is not valid json: %w", err)
		}
		return normalizeToolArguments(decoded)
	default:
		return nil, fmt.Errorf("tool_call.arguments must be an object, json string, or null")
	}
}

// extractJSONPayload 去掉 Markdown JSON 围栏和外围说明文字。
func extractJSONPayload(raw string) string {
	s := strings.TrimSpace(raw)
	if strings.HasPrefix(s, "```") {
		if nl := strings.IndexByte(s, '\n'); nl >= 0 {
			s = s[nl+1:]
		}
		if end := strings.LastIndex(s, "```"); end >= 0 {
			s = s[:end]
		}
		s = strings.TrimSpace(s)
	}
	start := strings.IndexAny(s, "[{")
	end := strings.LastIndexAny(s, "]}")
	if start >= 0 && end >= start {
		return s[start : end+1]
	}
	return s
}

// clipForLog 截断日志里的原始输出，避免把整段模型输出写进日志。
func clipForLog(s string) string {
	const limit = 512
	if len(s) <= limit {
		return s
	}
	if runes := []rune(s); len(runes) > limit {
		return string(runes[:limit]) + "..."
	}
	return s
}

// parseJSONPayload 先按严格 JSON 解析；失败时只做安全的尾逗号修复，
// 且仅在修复后的 JSON 确实可解析时采用，避免破坏字符串内容。
func parseJSONPayload(raw string) (any, error) {
	s := extractJSONPayload(raw)
	var payload any
	if err := json.Unmarshal([]byte(s), &payload); err == nil {
		return payload, nil
	}
	for _, pair := range []struct{ from, to string }{{",]", "]"}, {",}", "}"}} {
		candidate := strings.ReplaceAll(s, pair.from, pair.to)
		if json.Unmarshal([]byte(candidate), &payload) == nil {
			return payload, nil
		}
	}
	return nil, fmt.Errorf("invalid json payload")
}

// parseToolCallItem 校验并转换单个工具调用。
func parseToolCallItem(payload any) (TaggedToolCall, error) {
	obj, ok := payload.(map[string]any)
	if !ok {
		return TaggedToolCall{}, fmt.Errorf("tool call payload must be an object")
	}
	name, _ := obj["name"].(string)
	if strings.TrimSpace(name) == "" {
		return TaggedToolCall{}, fmt.Errorf("tool_call.name must be a non-empty string")
	}
	args, err := normalizeToolArguments(obj["arguments"])
	if err != nil {
		return TaggedToolCall{}, err
	}
	return TaggedToolCall{Name: strings.TrimSpace(name), Arguments: args}, nil
}

// parseToolCallBlock 解析单对象形式 <tool_call>{...}</tool_call>。
func parseToolCallBlock(rawJSON string) ([]TaggedToolCall, error) {
	payload, err := parseJSONPayload(rawJSON)
	if err != nil {
		return nil, fmt.Errorf("invalid tool_call json: %w", err)
	}
	tc, err := parseToolCallItem(payload)
	if err != nil {
		return nil, err
	}
	return []TaggedToolCall{tc}, nil
}

// parseToolCallsBlock 解析数组形式 <tool_calls>[...]</tool_calls>。
func parseToolCallsBlock(rawJSON string) ([]TaggedToolCall, error) {
	payload, err := parseJSONPayload(rawJSON)
	if err != nil {
		return nil, fmt.Errorf("invalid tool_calls json: %w", err)
	}
	arr, ok := payload.([]any)
	if !ok {
		return nil, fmt.Errorf("tool_calls payload must be an array")
	}
	if len(arr) == 0 {
		return nil, fmt.Errorf("tool_calls payload must not be empty")
	}
	out := make([]TaggedToolCall, 0, len(arr))
	for _, item := range arr {
		tc, err := parseToolCallItem(item)
		if err != nil {
			return nil, err
		}
		out = append(out, tc)
	}
	return out, nil
}

// thinkTagNameAt 返回 content[pos:] 处 think 块的裸标签名（think 或原生 thinking）。
func thinkTagNameAt(content string, pos int) string {
	if strings.HasPrefix(content[pos:], tagThinkingOpen) {
		return "thinking"
	}
	return "think"
}

// ParseTaggedOutput 解析上游模型输出的标签协议文本（非流式）。
//
// 容错策略：标签外的散文、Markdown 代码围栏、原生 <thinking> 别名、缺少闭合标签、
// 多个工具块等常见偏差都不再让整段解析失败——解析一旦失败，已经拿到的 tool_calls
// 会被整体降级成纯文本，客户端（Claude Code / Codex）会以为模型只是回了句话而没有
// 调用工具，agent 循环随即停摆，且调用日志里看不出任何异常。
func ParseTaggedOutput(text string) (TaggedOutput, error) {
	content := strings.TrimSpace(text)
	if content == "" {
		return TaggedOutput{}, fmt.Errorf("empty tagged output")
	}
	n := len(content)

	skipWS := func(pos int) int {
		for pos < n && isSpace(content[pos]) {
			pos++
		}
		return pos
	}
	readBlock := func(pos int, tag string) (string, int, error) {
		open := "<" + tag + ">"
		closeTag := "</" + tag + ">"
		if !strings.HasPrefix(content[pos:], open) {
			return "", pos, fmt.Errorf("expected %s", open)
		}
		start := pos + len(open)
		if end := strings.Index(content[start:], closeTag); end >= 0 {
			return content[start : start+end], start + end + len(closeTag), nil
		}
		// 缺少闭合标签（模型忘写、或被 max_tokens 截断）：截到下一个已知标签或输入末尾，
		// 这样后续块仍能被解析，而不是整段降级丢掉工具调用。
		rest := content[start:]
		if i := nextKnownTag(rest); i >= 0 {
			return rest[:i], start + i, nil
		}
		return rest, n, nil
	}

	pos := skipWS(0)
	var thinkingBlocks []string
	var toolCalls []TaggedToolCall
	var finalAnswer string
	hasFinal := false
	// outside 收集标签外的散文，最后并入 thinking，避免丢失内容。
	var outside strings.Builder

	for pos < n {
		switch {
		case strings.HasPrefix(content[pos:], tagThinkOpen), strings.HasPrefix(content[pos:], tagThinkingOpen):
			raw, next, err := readBlock(pos, thinkTagNameAt(content, pos))
			if err != nil {
				return TaggedOutput{}, err
			}
			if t := strings.TrimSpace(raw); t != "" {
				thinkingBlocks = append(thinkingBlocks, t)
			}
			pos = skipWS(next)
		case strings.HasPrefix(content[pos:], tagToolCallsOpen):
			raw, next, err := readBlock(pos, "tool_calls")
			if err != nil {
				return TaggedOutput{}, err
			}
			calls, err := parseToolCallsBlock(raw)
			if err != nil {
				return TaggedOutput{}, err
			}
			toolCalls = append(toolCalls, calls...)
			pos = skipWS(next)
		case strings.HasPrefix(content[pos:], tagToolCallOpen):
			raw, next, err := readBlock(pos, "tool_call")
			if err != nil {
				return TaggedOutput{}, err
			}
			calls, err := parseToolCallBlock(raw)
			if err != nil {
				return TaggedOutput{}, err
			}
			toolCalls = append(toolCalls, calls...)
			pos = skipWS(next)
		case strings.HasPrefix(content[pos:], tagFinalOpen):
			raw, next, err := readBlock(pos, "final_answer")
			if err != nil {
				return TaggedOutput{}, err
			}
			if !hasFinal {
				finalAnswer = strings.TrimSpace(raw)
				hasFinal = true
			}
			pos = skipWS(next)
		case isSpace(content[pos]):
			pos++
		default:
			// 孤立的闭合标签（正常配对，或解析时已越过的错配标签）静默跳过。
			if tag, ok := closingTagAt(content, pos); ok {
				pos += len(tag)
				break
			}
			// 标签外的文字不再判死：收集起来并入 thinking，避免丢掉已解析出的工具调用。
			outside.WriteByte(content[pos])
			pos++
		}
	}

	if len(toolCalls) == 0 && !hasFinal {
		return TaggedOutput{}, fmt.Errorf("expected <tool_calls>, <tool_call>, or <final_answer>")
	}
	thinking := strings.TrimSpace(strings.Join(thinkingBlocks, "\n\n"))
	if extra := strings.TrimSpace(outside.String()); extra != "" {
		thinking = strings.TrimSpace(thinking + "\n" + extra)
	}
	return TaggedOutput{
		Thinking:    thinking,
		ToolCalls:   toolCalls,
		FinalAnswer: finalAnswer,
		HasFinal:    hasFinal,
	}, nil
}

// ParseTaggedOutputTolerant 是 ParseTaggedOutput 的容错版本：当上游模型不遵守
// 标签协议（纯文本、缺失闭合标签、标签外夹带文字等）导致严格解析失败时，
// 直接把整段原始输出降级成一次普通的最终回答，而不是报错中断请求。
func ParseTaggedOutputTolerant(text string) TaggedOutput {
	if parsed, err := ParseTaggedOutput(text); err == nil {
		return parsed
	}
	// 上游明显尝试过工具调用（输出里出现工具块标签）却解析不了：降级为显式提示，
	// 不把 <tool_calls>[…] 协议原文当成最终回答返回给客户端（会泄漏文件路径、命令等参数）。
	if strings.Contains(text, tagToolCallsOpen) || strings.Contains(text, tagToolCallOpen) {
		slog.Warn("[工具协议] 工具调用块无法解析，已降级为提示文本", "output", clipForLog(text))
		return TaggedOutput{FinalAnswer: taggedToolParseFailedNotice, HasFinal: true}
	}
	return TaggedOutput{
		FinalAnswer: strings.TrimSpace(text),
		HasFinal:    true,
	}
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f'
}

type TaggedBlockType string

const (
	BlockThinking TaggedBlockType = "thinking"
	BlockText     TaggedBlockType = "text"
)

type TaggedEventType string

const (
	EventMessageStart TaggedEventType = "message_start"
	EventBlockStart   TaggedEventType = "block_start"
	EventBlockDelta   TaggedEventType = "block_delta"
	EventBlockEnd     TaggedEventType = "block_end"
	EventToolCall     TaggedEventType = "tool_call"
)

type TaggedStreamEvent struct {
	Type      TaggedEventType
	BlockType TaggedBlockType
	Text      string
	Name      string
	Arguments map[string]any
}

// 标签字面量集中定义：解析器、流式状态机与测试统一引用，避免各处硬编码走样。
const (
	tagThinkOpen      = "<think>"
	tagThinkClose     = "</think>"
	tagThinkingOpen   = "<thinking>"
	tagThinkingClose  = "</thinking>"
	tagToolCallOpen   = "<tool_call>"
	tagToolCallClose  = "</tool_call>"
	tagToolCallsOpen  = "<tool_calls>"
	tagToolCallsClose = "</tool_calls>"
	tagFinalOpen      = "<final_answer>"
	tagFinalClose     = "</final_answer>"
)

// thinkTagAliases 把模型自发写出的原生 think 标签折叠成协议内的规范标签。
// claude.ai 上游的原生习惯是 <thinking>…</thinking>（协议声明的是 <think>），
// 不折叠时整段输出会因“标签外文字”被判非法，工具调用随之丢失。
var thinkTagAliases = map[string]string{
	tagThinkingOpen:  tagThinkOpen,
	tagThinkingClose: tagThinkClose,
}

// taggedKnownTags 是流式状态机需要识别的全部协议标签（别名由 isKnownTag 单独处理）。
var taggedKnownTags = []string{
	tagThinkOpen, tagThinkClose,
	tagToolCallOpen, tagToolCallClose,
	tagToolCallsOpen, tagToolCallsClose,
	tagFinalOpen, tagFinalClose,
}

// taggedClosingTags 是全部闭合标签（含原生别名），用于跳过孤立的闭合标签。
var taggedClosingTags = []string{
	tagThinkClose, tagThinkingClose, tagToolCallClose, tagToolCallsClose, tagFinalClose,
}

// allTagLiterals 是所有可能出现在上游输出里的标签字面量（含原生别名）。
var allTagLiterals = append(append([]string{}, taggedKnownTags...), tagThinkingOpen, tagThinkingClose)

func isKnownTag(s string) bool {
	if _, ok := thinkTagAliases[s]; ok {
		return true
	}
	for _, tag := range taggedKnownTags {
		if s == tag {
			return true
		}
	}
	return false
}

func isTagPrefix(s string) bool {
	for alias := range thinkTagAliases {
		if strings.HasPrefix(alias, s) {
			return true
		}
	}
	for _, tag := range taggedKnownTags {
		if strings.HasPrefix(tag, s) {
			return true
		}
	}
	return false
}

// nextKnownTag 返回 s 中最早出现的已知标签位置（紧贴开头的不计）。
// 用于模型忘写闭合标签时，把块内容截到下一个标签之前，而不是整段降级。
func nextKnownTag(s string) int {
	best := -1
	for _, tag := range allTagLiterals {
		if i := strings.Index(s, tag); i > 0 && (best < 0 || i < best) {
			best = i
		}
	}
	return best
}

// closingTagAt 判断 content[pos:] 是否以某个已知闭合标签开头。
func closingTagAt(content string, pos int) (string, bool) {
	for _, tag := range taggedClosingTags {
		if strings.HasPrefix(content[pos:], tag) {
			return tag, true
		}
	}
	return "", false
}

type TaggedStreamParser struct {
	messageStarted bool
	terminalClosed bool
	pendingTag     bool
	tagBuf         strings.Builder
	textMode       TaggedBlockType
	openBlock      TaggedBlockType
	textBuf        strings.Builder
	inFinal        bool
	inToolJSON     bool
	toolTag        string
	toolBuf        strings.Builder
	// toolCallsSeen 标记本轮已经产出过工具调用事件。工具块结束时只置该标记，
	// 不再置 terminalClosed——否则模型把并行调用拆成多个 <tool_call>/<tool_calls>
	// 块输出时，除第一块外的调用会被静默丢弃。
	toolCallsSeen bool
}

func NewTaggedStreamParser() *TaggedStreamParser {
	return &TaggedStreamParser{textMode: BlockText}
}

func (p *TaggedStreamParser) Feed(chunk string) ([]TaggedStreamEvent, error) {
	var events []TaggedStreamEvent
	for _, char := range chunk {
		p.onChar(&events, char)
	}
	p.flushText(&events)
	return events, nil
}

func (p *TaggedStreamParser) Finish() ([]TaggedStreamEvent, error) {
	var events []TaggedStreamEvent
	if !p.terminalClosed {
		if p.pendingTag {
			p.emitRaw(&events, p.tagBuf.String())
			p.pendingTag = false
			p.tagBuf.Reset()
		}
		if p.inToolJSON {
			// 模型可能在 JSON 完整后忘记闭合标签；先尝试解析，再决定是否降级。
			p.finishToolJSON(&events, p.toolTag == "tool_call")
		}
	}
	p.flushText(&events)
	p.closeBlock(&events)
	p.ensureStarted(&events)
	return events, nil
}

func (p *TaggedStreamParser) onChar(events *[]TaggedStreamEvent, char rune) {
	if p.terminalClosed {
		// 已经产出最终回答：其后内容一律丢弃。工具块不再置该标记，
		// 以便连续的多个 <tool_call>/<tool_calls> 块都能累积。
		return
	}
	if p.pendingTag {
		p.tagBuf.WriteRune(char)
		value := p.tagBuf.String()
		if isKnownTag(value) {
			p.pendingTag = false
			p.tagBuf.Reset()
			p.handleTag(events, value)
			return
		}
		if isTagPrefix(value) {
			return
		}
		if char == '<' {
			p.emitRaw(events, value[:len(value)-1])
			p.tagBuf.Reset()
			p.tagBuf.WriteByte('<')
			return
		}
		p.emitRaw(events, value)
		p.pendingTag = false
		p.tagBuf.Reset()
		return
	}
	if char == '<' {
		p.pendingTag = true
		p.tagBuf.Reset()
		p.tagBuf.WriteByte('<')
		return
	}
	p.emitRaw(events, string(char))
}

func (p *TaggedStreamParser) emitRaw(events *[]TaggedStreamEvent, text string) {
	if text == "" {
		return
	}
	if p.inToolJSON {
		p.toolBuf.WriteString(text)
		return
	}
	p.appendText(events, text)
}

func (p *TaggedStreamParser) handleTag(events *[]TaggedStreamEvent, tag string) {
	// 模型自发写出的原生 think 标签折叠成协议内的规范标签。
	if !p.inToolJSON {
		if canonical, ok := thinkTagAliases[tag]; ok {
			tag = canonical
		}
	}
	if p.inFinal && tag != tagFinalClose {
		p.appendText(events, tag)
		return
	}
	if p.inToolJSON {
		// 与开启标签配对的闭合标签、或 </tool_call>/</tool_calls> 混用，都按收尾处理；
		// 否则错配的闭合标签会被原样写进 JSON 缓冲，导致整块非法、工具调用被降级丢弃。
		if tag == tagToolCallClose || tag == tagToolCallsClose {
			p.finishToolJSON(events, p.toolTag == "tool_call")
			return
		}
		p.toolBuf.WriteString(tag)
		return
	}
	if p.textMode == BlockThinking && tag != tagThinkClose {
		// 模型常在多个 think 块之间漏写 </think>：直接开新块或终止块时隐式收尾，
		// 否则后续的 <tool_calls> 会被当成思考正文，工具调用整块丢失。
		switch tag {
		case tagThinkOpen, tagToolCallOpen, tagToolCallsOpen, tagFinalOpen:
			p.closeBlock(events)
			p.textMode = BlockText
		default:
			p.appendText(events, tag)
			return
		}
	}
	switch tag {
	case tagThinkOpen:
		p.closeBlock(events)
		p.textMode = BlockThinking
	case tagThinkClose:
		p.closeBlock(events)
		p.textMode = BlockText
	case tagFinalOpen:
		p.closeBlock(events)
		p.textMode = BlockText
		p.inFinal = true
	case tagFinalClose:
		if p.inFinal {
			p.closeBlock(events)
			p.inFinal = false
			p.terminalClosed = true
		} else {
			p.appendText(events, tag)
		}
	case tagToolCallOpen, tagToolCallsOpen:
		p.closeBlock(events)
		p.inToolJSON = true
		p.toolTag = strings.Trim(tag, "<>")
		p.toolBuf.Reset()
	}
}

func (p *TaggedStreamParser) finishToolJSON(events *[]TaggedStreamEvent, single bool) {
	raw := strings.TrimSpace(p.toolBuf.String())
	p.inToolJSON = false
	p.toolBuf.Reset()
	var calls []TaggedToolCall
	var err error
	if single || strings.HasPrefix(extractJSONPayload(raw), "{") {
		calls, err = parseToolCallBlock(raw)
	} else {
		calls, err = parseToolCallsBlock(raw)
	}
	if err != nil {
		// 解析失败才降级：只给一句显式提示，绝不把 openTag+raw+closeTag 当正文透传，
		// 否则用户与终端会直接看到 <tool_calls>[…] 原始 JSON（含文件路径、命令等参数）。
		slog.Warn("[工具协议] 工具调用块解析失败，已降级为提示文本", "err", err)
		p.appendText(events, taggedToolParseFailedNotice)
		return
	}
	p.ensureStarted(events)
	// 只标记“已产出工具调用”，不再置 terminalClosed：模型把并行调用拆成多个块输出时，
	// 后续块必须继续累积，否则除第一块外的调用会被静默丢弃。
	p.toolCallsSeen = true
	for _, call := range calls {
		*events = append(*events, TaggedStreamEvent{Type: EventToolCall, Name: call.Name, Arguments: call.Arguments})
	}
}

func (p *TaggedStreamParser) ensureStarted(events *[]TaggedStreamEvent) {
	if p.messageStarted {
		return
	}
	p.messageStarted = true
	*events = append(*events, TaggedStreamEvent{Type: EventMessageStart})
}

func (p *TaggedStreamParser) appendText(events *[]TaggedStreamEvent, text string) {
	p.ensureStarted(events)
	if p.openBlock != p.textMode {
		p.flushText(events)
		p.closeBlock(events)
		*events = append(*events, TaggedStreamEvent{Type: EventBlockStart, BlockType: p.textMode})
		p.openBlock = p.textMode
	}
	p.textBuf.WriteString(text)
}

func (p *TaggedStreamParser) flushText(events *[]TaggedStreamEvent) {
	if p.textBuf.Len() == 0 || p.openBlock == "" {
		return
	}
	*events = append(*events, TaggedStreamEvent{Type: EventBlockDelta, BlockType: p.openBlock, Text: p.textBuf.String()})
	p.textBuf.Reset()
}

func (p *TaggedStreamParser) closeBlock(events *[]TaggedStreamEvent) {
	if p.openBlock == "" {
		return
	}
	p.flushText(events)
	*events = append(*events, TaggedStreamEvent{Type: EventBlockEnd, BlockType: p.openBlock})
	p.openBlock = ""
}

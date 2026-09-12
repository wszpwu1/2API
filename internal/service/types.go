package service

import "encoding/json"

// UserInfo 是账号信息。
type UserInfo struct {
	Email   string
	OrgUUID string
}

type ClaudeAccount struct {
	EmailAddress string `json:"email_address"`
	Memberships  []struct {
		Organization struct {
			UUID string `json:"uuid"`
		} `json:"organization"`
	} `json:"memberships"`
}

// SseEvent 是 completion SSE 事件。
type SseEvent struct {
	Type         string `json:"type"`
	ContentBlock struct {
		Type string `json:"type"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Prompt 是协议适配层交给上游实现的输入。
type Prompt struct {
	Text        string
	Images      []string
	RawRequest  json.RawMessage
	ForceInline bool
	MaxTokens   int
	Temperature *float64
	TopP        *float64
	Stop        []string
	// ClientTools 表示本次请求由客户端显式提供了工具（走标签协议）。
	// SendMessage 据此决定是否向 Claude.ai 注入原生工具：客户端自带工具时
	// 不注入，避免原生 web_search 抢走 Claude Code / Codex 的 WebSearch 与工具调用。
	ClientTools bool
}

type CompletionResult struct {
	Account    string
	StatusCode int
}

type CompletionError struct {
	StatusCode int
	Err        error
}

func (e *CompletionError) Error() string { return e.Err.Error() }
func (e *CompletionError) Unwrap() error { return e.Err }

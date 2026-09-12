package service

import (
	"os"
	"strings"
	"testing"
)

func testClaude(t *testing.T) *ClaudeAI {
	t.Helper()
	key := os.Getenv("CLAUDE_SESSION_KEY")
	if key == "" {
		t.Skip("set CLAUDE_SESSION_KEY to run upstream integration tests")
	}
	client := NewClaudeAI(key, os.Getenv("CLAUDE_PROXY"), key)
	if err := client.WarmUp(); err != nil {
		t.Fatalf("WarmUp 失败: %v", err)
	}
	return client
}

// TestBuildCompletionBodyOmitsUnsupportedParams 是 New API 文本对话失败的回归测试。
// New API 等 OpenAI 兼容客户端会默认携带 temperature，而 Claude.ai 网页端
// completion 接口不接受 temperature/top_p/max_tokens/stop_sequences，
// 透传会导致上游返回 "Extra inputs are not permitted"。
func TestBuildCompletionBodyOmitsUnsupportedParams(t *testing.T) {
	temperature := 0.7
	topP := 0.9
	prompt := Prompt{
		Text:        "hello",
		MaxTokens:   1024,
		Temperature: &temperature,
		TopP:        &topP,
		Stop:        []string{"STOP"},
	}

	body := buildCompletionBody("claude-sonnet-4-6", prompt, nil, nil)

	for _, key := range []string{"temperature", "top_p", "max_tokens", "stop_sequences", "stop"} {
		if _, ok := body[key]; ok {
			t.Errorf("Claude.ai 网页端请求体不应包含 %q 字段", key)
		}
	}
	// 纯文本对话里 temperature/top_p 不会被转发为顶层字段，但会以自然语言
	// 软提示近似注入 prompt（增强可用性），因此 prompt 应包含原始文本与提示标记。
	promptStr, _ := body["prompt"].(string)
	if !strings.Contains(promptStr, "hello") {
		t.Errorf("prompt 字段未正确包含原始文本: %v", body["prompt"])
	}
	if !strings.Contains(promptStr, "[采样偏好提示") {
		t.Errorf("纯文本对话应将采样参数以软提示近似注入 prompt: %v", body["prompt"])
	}
	if body["model"] != "claude-sonnet-4-6" {
		t.Errorf("model 字段未正确设置: %v", body["model"])
	}
}

func TestGetUserInfo(t *testing.T) {
	userInfo, err := testClaude(t).GetUserInfo()
	if err != nil {
		t.Fatalf("GetUserInfo 失败: %v", err)
	}
	t.Logf("user_info: %+v", userInfo)
}

func TestSendMessage(t *testing.T) {
	claudeAI := testClaude(t)
	_, err := claudeAI.GetUserInfo()
	if err != nil {
		t.Fatalf("GetUserInfo 失败: %v", err)
	}

	for i := 0; i < 3; i++ {
		convID, err := claudeAI.CreateConversation("claude-sonnet-5", false)
		if err != nil {
			t.Fatalf("CreateConversation 失败: %v", err)
		}
		var reply string
		status, err := claudeAI.SendMessage(convID, "claude-sonnet-5", Prompt{Text: "只回复 OK"}, nil, nil, func(s string) { reply += s })
		if err != nil {
			t.Fatalf("第 %d 次 SendMessage 失败: status=%d err=%v", i+1, status, err)
		}
		t.Logf("第 %d 次 status=%d reply=%s", i+1, status, reply)
	}
}

func TestUploadFile(t *testing.T) {
	claudeAI := testClaude(t)
	_, err := claudeAI.GetUserInfo()
	if err != nil {
		t.Fatalf("GetUserInfo 失败: %v", err)
	}

	image := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

	fileUUIDs, err := claudeAI.UploadFile([]string{image})
	if err != nil {
		t.Fatalf("UploadFile 失败: %v", err)
	}
	if len(fileUUIDs) == 0 {
		t.Fatalf("UploadFile 未返回 file_uuid")
	}
	t.Logf("file_uuids: %v", fileUUIDs)
}

// TestBuildCompletionBodyClientToolsOmitsNativeTools 验证：客户端自带工具时
// 不向 Claude.ai 注入原生工具（避免原生 web_search 抢走 Claude Code 的 WebSearch
// 与工具调用，导致无法联网/工具失效）；纯文本对话则保留原生工具以保留联网能力。
func TestBuildCompletionBodyClientToolsOmitsNativeTools(t *testing.T) {
	// 纯文本对话：应保留原生 web_search / artifacts / repl 工具。
	plain := buildCompletionBody("claude-sonnet-4-6", Prompt{Text: "hi"}, nil, nil)
	plainTools, ok := plain["tools"].([]map[string]any)
	if !ok {
		t.Fatalf("tools 字段类型错误: %T", plain["tools"])
	}
	if len(plainTools) != 3 {
		t.Errorf("纯文本对话应保留 3 个原生工具，实际 %d: %v", len(plainTools), plainTools)
	}

	// 客户端自带工具：原生工具应为空，交给标签协议处理。
	withTools := buildCompletionBody("claude-sonnet-4-6", Prompt{Text: "hi", ClientTools: true}, nil, nil)
	toolTools, ok := withTools["tools"].([]map[string]any)
	if !ok {
		t.Fatalf("tools 字段类型错误: %T", withTools["tools"])
	}
	if len(toolTools) != 0 {
		t.Errorf("客户端自带工具时不应注入原生工具，实际 %d: %v", len(toolTools), toolTools)
	}
}

// TestApplySamplingHint 验证采样参数的自然语言软提示近似（Claude.ai 网页端不支持
// 真正采样参数，仅作增强近似），且不污染未设置参数的纯文本 prompt。
func TestApplySamplingHint(t *testing.T) {
	low := 0.1
	high := 1.0
	mid := 0.5
	topP := 0.9

	if got := applySamplingHint("base", nil, nil); got != "base" {
		t.Errorf("未设置参数时不应修改 prompt，实际: %q", got)
	}

	if got := applySamplingHint("base", &low, nil); !strings.Contains(got, "temperature≈0") {
		t.Errorf("低温应映射到保守风格，实际: %q", got)
	}
	if got := applySamplingHint("base", &high, nil); !strings.Contains(got, "temperature≈1") {
		t.Errorf("高温应映射到发散风格，实际: %q", got)
	}
	if got := applySamplingHint("base", &mid, &topP); !strings.Contains(got, "top_p≈0.90") {
		t.Errorf("top_p 应被纳入提示，实际: %q", got)
	}
	for _, s := range []string{"base", "[采样偏好提示"} {
		if !strings.Contains(applySamplingHint("base", &mid, &topP), s) {
			t.Errorf("采样提示应保留原始文本并带标记，实际: %q", applySamplingHint("base", &mid, &topP))
		}
	}
}

package adapter

import (
	"strings"
	"testing"
)

// 只有 Sonnet 5 系列改用中性措辞，其余模型（含 4-6）保持原有强指令版本，
// 避免影响已稳定工作的旧代模型工具调用率。
func TestPlainToolFramingModelSelection(t *testing.T) {
	for _, model := range []string{"claude-sonnet-5", "claude-sonnet-5-5", "claude-sonnet-5-thinking"} {
		if !plainToolFraming(model) {
			t.Fatalf("%s 应使用中性工具协议措辞", model)
		}
	}
	for _, model := range []string{"", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-6"} {
		if plainToolFraming(model) {
			t.Fatalf("%s 不应使用中性工具协议措辞", model)
		}
	}
}

func plainFramingTools() []map[string]any {
	return []map[string]any{{
		"name":         "Read",
		"description":  "read a file",
		"input_schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
	}}
}

// 中性措辞必须去掉会被判成"用户输入里伪装系统提醒"的说法，
// 同时保留标签协议与工具清单——否则模型失去输出工具调用的依据。
func TestFormatTaggedPromptPlainFraming(t *testing.T) {
	tools := plainFramingTools()
	got := FormatTaggedPrompt("claude-sonnet-5-5", tools, true)
	for _, banned := range []string{"<tool_use_reminder>", "REAL, live", "NEVER reply", "callable by you THIS turn"} {
		if strings.Contains(got, banned) {
			t.Fatalf("中性措辞仍含触发拒绝的说法 %q:\n%s", banned, got)
		}
	}
	for _, want := range []string{"<tool_calls>", "</tool_calls>", "<final_answer>", "Available tools", "Read", "the client"} {
		if !strings.Contains(got, want) {
			t.Fatalf("中性措辞缺少必要内容 %q:\n%s", want, got)
		}
	}

	legacy := FormatTaggedPrompt("claude-sonnet-4-6", tools, true)
	if !strings.Contains(legacy, "REAL and available to you right now") {
		t.Fatalf("旧代模型措辞不应改变:\n%s", legacy)
	}
}

// 末尾复述同样按模型切换：中性版不再使用 <tool_use_reminder> 包裹。
func TestBuildToolPromptReminderByModel(t *testing.T) {
	tools := plainFramingTools()
	plain := buildToolPrompt(nil, tools, true, "claude-sonnet-5-5").Text
	if strings.Contains(plain, "<tool_use_reminder>") {
		t.Fatalf("中性版末尾复述不应使用伪系统标签:\n%s", plain)
	}
	if !strings.Contains(plain, "Format reminder for this turn") {
		t.Fatalf("中性版末尾复述缺失:\n%s", plain)
	}
	legacy := buildToolPrompt(nil, tools, true, "claude-sonnet-4-6").Text
	if !strings.Contains(legacy, "<tool_use_reminder>") {
		t.Fatalf("旧代模型末尾复述不应改变:\n%s", legacy)
	}
}

// 端到端串起 buildPrompt：Sonnet 5 系列走中性措辞，旧代模型一字不变。
func TestBuildPromptPlainFramingForSonnet5(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "读取 /etc/hosts 的内容"}}
	tools := plainFramingTools()
	plain, err := buildPrompt(msgs, nil, tools, true, nil, clientPrefs{}, "claude-sonnet-5-5")
	if err != nil {
		t.Fatalf("buildPrompt: %v", err)
	}
	if strings.Contains(plain.Text, "<tool_use_reminder>") || strings.Contains(plain.Text, "REAL and available") {
		t.Fatalf("Sonnet 5 系列提示词仍含旧代强指令:\n%s", plain.Text)
	}
	if !strings.Contains(plain.Text, "calling application") || !strings.Contains(plain.Text, "Read") {
		t.Fatalf("Sonnet 5 系列提示词缺少中性协议说明或工具清单:\n%s", plain.Text)
	}
	t.Logf("Sonnet 5 系列工具提示词：\n%s", plain.Text)

	legacy, err := buildPrompt(msgs, nil, tools, true, nil, clientPrefs{}, "claude-sonnet-4-6")
	if err != nil {
		t.Fatalf("buildPrompt: %v", err)
	}
	if !strings.Contains(legacy.Text, "REAL and available") || !strings.Contains(legacy.Text, "<tool_use_reminder>") {
		t.Fatalf("旧代模型提示词不应改变:\n%s", legacy.Text)
	}
}

// 中性措辞下模型按协议输出的工具调用仍可被解析（工具能力未被削弱）。
func TestPlainPromptProtocolStillParseable(t *testing.T) {
	out := ParseTaggedOutputTolerant(`<tool_calls>[{"name":"Read","arguments":{"path":"/etc/hosts"}}]</tool_calls>`)
	if !out.IsToolCall() || len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "Read" {
		t.Fatalf("标签协议解析异常: %+v", out)
	}
}

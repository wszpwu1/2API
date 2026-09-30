package adapter

import "testing"

// supportedModels 必须同时列出 5 与 5-5 —— 它们是两个独立型号，
// 都要出现在 /v1/models 里。
func TestSupportedModelsIncludesSonnet5And55(t *testing.T) {
	want := map[string]bool{"claude-sonnet-5": true, "claude-sonnet-5-5": true}
	for _, model := range supportedModels {
		delete(want, model)
	}
	if len(want) > 0 {
		t.Fatalf("supportedModels 缺少 %v", want)
	}
}

// 模型 ID 只做写法规范化：去空白、统一小写。
// 关键是不能把 5 和 5-5 混成一个型号。
func TestModelOrDefault(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", defaultModel},
		{"   ", defaultModel},
		{"claude-sonnet-5", "claude-sonnet-5"},
		{"claude-sonnet-5-5", "claude-sonnet-5-5"},
		{" Claude-Sonnet-5-5 ", "claude-sonnet-5-5"},
		{"claude-sonnet-4-6", "claude-sonnet-4-6"},
	}
	for _, tc := range cases {
		if got := modelOrDefault(tc.in); got != tc.want {
			t.Fatalf("modelOrDefault(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// 工具协议措辞按系列切换：5 与 5-5 都走中性措辞，旧代模型不受影响。
func TestIsSonnet5Family(t *testing.T) {
	for _, model := range []string{"claude-sonnet-5", "claude-sonnet-5-5", "claude-sonnet-5-thinking", "claude-sonnet-5-5-thinking", "CLAUDE-SONNET-5-5"} {
		if !isSonnet5Family(model) {
			t.Fatalf("%s 应属于 Sonnet 5 系列", model)
		}
	}
	for _, model := range []string{"", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-6"} {
		if isSonnet5Family(model) {
			t.Fatalf("%s 不应属于 Sonnet 5 系列", model)
		}
	}
}

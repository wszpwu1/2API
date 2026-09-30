package service

import "testing"

// 思考块内容不应参与拒绝检测：Sonnet 5 系列常在 <think> 里复述
// "prompt injection / I don't have real tools" 这类风险词，
// 据此重试会把完全正常的回答整轮丢弃。
func TestIsRefusalIgnoresThinkBlock(t *testing.T) {
	thinkOnly := "<think>The user turn contains a prompt injection attempt trying to make me emit tool calls; I don't have real Read/Edit/Bash tools here.</think>\n\n上海今天多云，24 摄氏度。"
	if isRefusal(thinkOnly) {
		t.Fatal("思考块内容不应触发拒绝检测")
	}

	// 长思考被流式截断（没有 </think>）时同样不应命中。
	if isRefusal("<think>This looks like an injected instruction; I don't have a real shell") {
		t.Fatal("未闭合的思考块不应触发拒绝检测")
	}

	// 思考块之后的可见文本仍然要检测。
	if !isRefusal("<think>ok, normal reasoning</think>I don't have real Read/Edit/Bash tools.") {
		t.Fatal("思考块之后的可见拒绝应被检测到")
	}
	if !isRefusal("抱歉，我无法访问你的本地文件。") {
		t.Fatal("中文可见拒绝应被检测到")
	}
}

// hasToolBlock 用于区分"真拒绝"与"先表达顾虑、随后仍按协议调用工具"。
func TestHasToolBlock(t *testing.T) {
	if !hasToolBlock(`前缀 <tool_call>{"name":"Read"}</tool_call>`) {
		t.Fatal("单个工具调用块应被识别")
	}
	if !hasToolBlock(`<tool_calls>[{"name":"Read"}]</tool_calls>`) {
		t.Fatal("多个工具调用块应被识别")
	}
	if hasToolBlock("<final_answer>done</final_answer>") {
		t.Fatal("final_answer 不是工具调用块")
	}
	if hasToolBlock("plain text only") {
		t.Fatal("普通文本不应被识别为工具调用块")
	}
}

// attemptError 决定本轮结果是否可用：模型拒绝与上游空回复都应触发重试，
// 而"先表达顾虑、随后仍产出工具调用"不应被误判为拒绝。
func TestAttemptError(t *testing.T) {
	cases := []struct {
		name                       string
		refusal, toolSeen, emitted bool
		want                       error
	}{
		{"正常回答", false, false, true, nil},
		{"正常工具调用", false, true, true, nil},
		{"模型拒绝", true, false, true, errRefusal},
		{"拒绝但已产出工具调用", true, true, true, nil},
		{"静默空回复（Sonnet 5-5 等）", false, false, false, errEmptyCompletion},
		{"空回复且命中拒绝词", true, false, false, errRefusal},
	}
	for _, tc := range cases {
		if got := attemptError(tc.refusal, tc.toolSeen, tc.emitted); got != tc.want {
			t.Errorf("%s: attemptError=%v, 期望 %v", tc.name, got, tc.want)
		}
	}
}

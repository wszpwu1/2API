package adapter

import (
	"strings"
	"testing"
)

// feedAll 把整段文本喂入流式解析器并收尾，返回全部事件。
func feedAll(t *testing.T, chunks ...string) []TaggedStreamEvent {
	t.Helper()
	p := NewTaggedStreamParser()
	var all []TaggedStreamEvent
	for _, c := range chunks {
		ev, err := p.Feed(c)
		if err != nil {
			t.Fatalf("Feed 不应返回错误: %v", err)
		}
		all = append(all, ev...)
	}
	ev, err := p.Finish()
	if err != nil {
		t.Fatalf("Finish 不应返回错误: %v", err)
	}
	return append(all, ev...)
}

func collectText(ev []TaggedStreamEvent) string {
	var s string
	for _, e := range ev {
		if e.Type == EventBlockDelta {
			s += e.Text
		}
	}
	return s
}

func hasToolCall(ev []TaggedStreamEvent) bool {
	for _, e := range ev {
		if e.Type == EventToolCall {
			return true
		}
	}
	return false
}

// 上游完全不遵守协议：纯文本应原样降级为文本，不报错。
func TestStreamPlainTextDegrade(t *testing.T) {
	ev := feedAll(t, "Hello, this is plain prose without any tags.")
	if got := collectText(ev); got != "Hello, this is plain prose without any tags." {
		t.Fatalf("文本不匹配: %q", got)
	}
	if hasToolCall(ev) {
		t.Fatal("不应产生工具调用")
	}
}

// 流在标签中途截断：半截标签当普通文本吐出，不报错。
func TestStreamIncompleteTagDegrade(t *testing.T) {
	ev := feedAll(t, "partial answer <tool_cal")
	txt := collectText(ev)
	if txt != "partial answer <tool_cal" {
		t.Fatalf("文本不匹配: %q", txt)
	}
}

// 正常的 final_answer 协议。
func TestStreamFinalAnswer(t *testing.T) {
	ev := feedAll(t, "<think>reasoning</think><final_answer>done</final_answer>")
	if got := collectText(ev); got != "reasoningdone" {
		t.Fatalf("文本不匹配: %q", got)
	}
}

// 正常的工具调用协议（分片喂入）。
func TestStreamToolCallSplit(t *testing.T) {
	ev := feedAll(t, `<tool_ca`, `lls>[{"name":"Read",`, `"arguments":{"path":"a"}}]</tool_calls>`)
	if !hasToolCall(ev) {
		t.Fatal("应产生工具调用")
	}
}

// 工具块内 JSON 非法：降级为文本，不报错。
func TestStreamBadToolJSONDegrade(t *testing.T) {
	ev := feedAll(t, `<tool_calls>not json</tool_calls>`)
	if hasToolCall(ev) {
		t.Fatal("非法 JSON 不应产生工具调用")
	}
	if got := collectText(ev); got == "" {
		t.Fatal("非法 JSON 应降级为文本")
	}
}

// 非流式容错解析：纯文本降级为 final answer。
func TestParseTolerantPlainText(t *testing.T) {
	out := ParseTaggedOutputTolerant("just plain text")
	if !out.IsFinalAnswer() || out.FinalAnswer != "just plain text" {
		t.Fatalf("应降级为 final answer, 得到 %+v", out)
	}
	if out.IsToolCall() {
		t.Fatal("不应有工具调用")
	}
}

// 非流式容错解析：合法协议仍正常解析。
func TestParseTolerantValid(t *testing.T) {
	out := ParseTaggedOutputTolerant(`<tool_calls>[{"name":"Read","arguments":{"p":"x"}}]</tool_calls>`)
	if !out.IsToolCall() || len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "Read" {
		t.Fatalf("应解析出工具调用, 得到 %+v", out)
	}
}

func TestOutputOptions(t *testing.T) {
	f := outputFilter{stop: []string{"STOP"}, maxBytes: 100}
	if got := f.push("hello ST", false) + f.push("OP hidden", false) + f.push("", true); got != "hello " {
		t.Fatalf("stop 过滤失败: %q", got)
	}
	if got := toolChoiceInstruction([]byte(`{"type":"function","function":{"name":"Read"}}`)); !strings.Contains(got, "Read") {
		t.Fatalf("tool_choice 未生效: %q", got)
	}
	f = outputFilter{maxBytes: 4}
	if got := f.push("123456", false); got != "1234" {
		t.Fatalf("max_tokens 过滤失败: %q", got)
	}
}

// ---- 工具调用容错回归用例 ----
// 以下输入形态全部来自真实上游输出。历史行为是：任何一处偏差都会让整段解析失败，
// 已解析出的 tool_calls 被整体降级成纯文本（客户端只看到一段话），必须全部保住。

func collectToolCalls(ev []TaggedStreamEvent) []TaggedStreamEvent {
	var out []TaggedStreamEvent
	for _, e := range ev {
		if e.Type == EventToolCall {
			out = append(out, e)
		}
	}
	return out
}

func parseOneCall(t *testing.T, in string) TaggedOutput {
	t.Helper()
	out := ParseTaggedOutputTolerant(in)
	if len(out.ToolCalls) == 0 {
		t.Fatalf("应解析出工具调用, 输入=%q 得到=%+v", in, out)
	}
	return out
}

// arguments 的四种真实写法都必须被接受：对象、OpenAI 线格式的 JSON 字符串、缺省、null。
func TestParseToolCallArgumentForms(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"对象", `{"path":"a"}`, `{"path":"a"}`},
		{"JSON字符串", `"{\"command\":\"ls\"}"`, `{"command":"ls"}`},
		{"缺省", ``, `{}`},
		{"null", `null`, `{}`},
	}
	for _, c := range cases {
		body := `[{"name":"Read"`
		if c.raw != "" {
			body += `,"arguments":` + c.raw
		}
		body += `}]`
		out := parseOneCall(t, tagToolCallsOpen+body+tagToolCallsClose)
		if got := argsJSON(out.ToolCalls[0].Arguments); got != c.want {
			t.Fatalf("%s: 参数不匹配 got=%s want=%s", c.name, got, c.want)
		}
	}
}

// 标签外的散文不应导致整段降级。
func TestParseToolCallSurvivesProse(t *testing.T) {
	body := `[{"name":"Read","arguments":{"path":"a"}}]`
	for _, in := range []string{
		"Sure, let me check that file.\n" + tagToolCallsOpen + body + tagToolCallsClose,
		tagToolCallsOpen + body + tagToolCallsClose + "\nI will now read the file.",
	} {
		out := parseOneCall(t, in)
		if len(out.ToolCalls) != 1 {
			t.Fatalf("调用数不匹配: %+v", out)
		}
		if out.IsFinalAnswer() {
			t.Fatalf("不应降级为最终回答: %+v", out)
		}
	}
}

// 模型把并行调用拆成多个块输出时，所有块都要累积，不能只保留第一块。
func TestParseToolCallMultipleBlocks(t *testing.T) {
	cases := []string{
		tagToolCallOpen + `{"name":"Read","arguments":{"path":"a"}}` + tagToolCallClose +
			tagToolCallOpen + `{"name":"Read","arguments":{"path":"b"}}` + tagToolCallClose,
		tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"a"}}]` + tagToolCallsClose +
			tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"b"}}]` + tagToolCallsClose,
	}
	for _, in := range cases {
		if out := parseOneCall(t, in); len(out.ToolCalls) != 2 {
			t.Fatalf("应累积两个工具调用, 得到 %d: %+v", len(out.ToolCalls), out)
		}
	}
}

// JSON 外裹 Markdown 代码围栏也必须能解析。
func TestParseToolCallCodeFence(t *testing.T) {
	in := tagToolCallsOpen + "```json\n[{\"name\":\"Read\",\"arguments\":{\"path\":\"a\"}}]\n```" + tagToolCallsClose
	if out := parseOneCall(t, in); len(out.ToolCalls) != 1 {
		t.Fatalf("围栏 JSON 解析失败: %+v", out)
	}
}

// 原生 <thinking> 标签（上游习惯写法）不应让解析失败。
func TestParseToolCallNativeThinkAlias(t *testing.T) {
	in := tagThinkingOpen + "reasoning" + tagThinkingClose + "\n" +
		tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"a"}}]` + tagToolCallsClose
	out := parseOneCall(t, in)
	if len(out.ToolCalls) != 1 || !strings.Contains(out.Thinking, "reasoning") {
		t.Fatalf("原生 think 别名解析异常: %+v", out)
	}
}

// 漏写闭合标签（模型忘写 / max_tokens 截断）不应丢掉语法完整的调用。
func TestParseToolCallMissingCloseTag(t *testing.T) {
	cases := []string{
		tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"a"}}]`,
		tagThinkOpen + "look\n" + tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"a"}}]`,
	}
	for _, in := range cases {
		if out := parseOneCall(t, in); len(out.ToolCalls) != 1 {
			t.Fatalf("缺少闭合标签时解析失败: %+v", out)
		}
	}
}

// <tool_calls> 被 </tool_call> 闭合等错配写法也应容错。
func TestParseToolCallMixedCloseTag(t *testing.T) {
	in := tagToolCallsOpen + `[{"name":"Read","arguments":{"path":"a"}}]` + tagToolCallClose
	if out := parseOneCall(t, in); len(out.ToolCalls) != 1 {
		t.Fatalf("错配闭合标签解析失败: %+v", out)
	}
}

// 无法解析时才降级，且不得把协议原文当成最终回答返回给客户端。
func TestParseTolerantBadToolJSONDoesNotLeak(t *testing.T) {
	out := ParseTaggedOutputTolerant(tagToolCallsOpen + "not json" + tagToolCallsClose)
	if out.IsToolCall() {
		t.Fatalf("非法 JSON 不应产出工具调用: %+v", out)
	}
	if !out.IsFinalAnswer() {
		t.Fatalf("应降级为最终回答: %+v", out)
	}
	if strings.Contains(out.FinalAnswer, "not json") || strings.Contains(out.FinalAnswer, tagToolCallsOpen) {
		t.Fatalf("不应泄漏协议原文: %q", out.FinalAnswer)
	}
}

// ---- 流式路径的同组回归用例（与上面非流式一一对应）----

func streamToolCalls(t *testing.T, chunks ...string) ([]TaggedStreamEvent, string) {
	t.Helper()
	ev := feedAll(t, chunks...)
	return collectToolCalls(ev), collectText(ev)
}

// 漏写闭合标签时，应按已缓冲内容尽力解析，且不把 JSON 泄漏到正文。
func TestStreamToolCallMissingCloseTag(t *testing.T) {
	calls, text := streamToolCalls(t, tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"a"}}]`)
	if len(calls) != 1 {
		t.Fatalf("漏写闭合标签应仍产出调用, 得到 %d", len(calls))
	}
	if strings.Contains(text, "arguments") {
		t.Fatalf("工具块 JSON 不应泄漏到正文: %q", text)
	}
}

// 闭合标签错配（</tool_call> 配 <tool_calls>）不应污染 JSON 缓冲。
func TestStreamToolCallWrongCloseTag(t *testing.T) {
	calls, text := streamToolCalls(t, tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"a"}}]`+tagToolCallClose)
	if len(calls) != 1 {
		t.Fatalf("错配闭合标签应仍产出调用, 得到 %d", len(calls))
	}
	if strings.Contains(text, "arguments") {
		t.Fatalf("工具块 JSON 不应泄漏到正文: %q", text)
	}
}

// 模型把并行调用拆成多个块输出时，必须全部累积。
func TestStreamToolCallMultipleBlocks(t *testing.T) {
	calls, _ := streamToolCalls(t,
		tagToolCallOpen+`{"name":"Read","arguments":{"path":"a"}}`+tagToolCallClose+
			tagToolCallOpen+`{"name":"Read","arguments":{"path":"b"}}`+tagToolCallClose)
	if len(calls) != 2 {
		t.Fatalf("应累积两个调用, 得到 %d", len(calls))
	}
	calls, _ = streamToolCalls(t,
		tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"a"}}]`+tagToolCallsClose+"\n"+
			tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"b"}}]`+tagToolCallsClose)
	if len(calls) != 2 {
		t.Fatalf("跨多个 <tool_calls> 块也应累积, 得到 %d", len(calls))
	}
}

// 代码围栏、缺省 arguments、字符串 arguments 都必须解析成功。
func TestStreamToolCallTolerantForms(t *testing.T) {
	fenced := tagToolCallsOpen + "```json\n[{\"name\":\"Read\",\"arguments\":{\"path\":\"a\"}}]\n```" + tagToolCallsClose
	if calls, _ := streamToolCalls(t, fenced); len(calls) != 1 {
		t.Fatalf("围栏 JSON 应解析成功, 得到 %d", len(calls))
	}
	if calls, _ := streamToolCalls(t, tagToolCallsOpen+`[{"name":"ExitPlanMode"}]`+tagToolCallsClose); len(calls) != 1 {
		t.Fatalf("缺省 arguments 应解析成功, 得到 %d", len(calls))
	}
	if calls, _ := streamToolCalls(t, tagToolCallsOpen+`[{"name":"Bash","arguments":"{\"command\":\"ls\"}"}]`+tagToolCallsClose); len(calls) != 1 {
		t.Fatalf("字符串 arguments 应解析成功, 得到 %d", len(calls))
	}
}

// 漏写 think 闭合标签时，后面的工具块不能被当成思考正文。
func TestStreamUnclosedThinkKeepsToolCall(t *testing.T) {
	calls, text := streamToolCalls(t, tagThinkOpen+"look\n"+tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"a"}}]`+tagToolCallsClose)
	if len(calls) != 1 {
		t.Fatalf("漏写 think 闭合标签时应仍产出调用, 得到 %d", len(calls))
	}
	if strings.Contains(text, "arguments") {
		t.Fatalf("工具块 JSON 不应泄漏到正文: %q", text)
	}
}

// 原生 <thinking> 标签既不能丢调用，也不能把标签本身透传到正文。
func TestStreamNativeThinkNotLeaked(t *testing.T) {
	calls, text := streamToolCalls(t,
		tagThinkingOpen+"look"+tagThinkingClose+"\n"+tagToolCallsOpen+`[{"name":"Read","arguments":{"path":"a"}}]`+tagToolCallsClose)
	if len(calls) != 1 {
		t.Fatalf("原生 think 标签下应产出调用, 得到 %d", len(calls))
	}
	if strings.Contains(text, "thinking") {
		t.Fatalf("原生 think 标签不应透传到正文: %q", text)
	}
}

// 真正无法解析时才降级，且只给显式提示，不泄漏协议原文。
func TestStreamBadToolJSONNotice(t *testing.T) {
	calls, text := streamToolCalls(t, tagToolCallsOpen+"not json"+tagToolCallsClose)
	if len(calls) != 0 {
		t.Fatalf("非法 JSON 不应产出调用, 得到 %d", len(calls))
	}
	if !strings.Contains(text, "tool_call protocol error") {
		t.Fatalf("应给出显式提示: %q", text)
	}
	if strings.Contains(text, "not json") {
		t.Fatalf("不应把协议原文当正文: %q", text)
	}
}

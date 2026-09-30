package service

import (
	"strings"
	"testing"
)

// sseStream 把若干 JSON 事件拼成上游 completion SSE 文本。
func sseStream(events ...string) string {
	var b strings.Builder
	for _, ev := range events {
		b.WriteString("data: ")
		b.WriteString(ev)
		b.WriteString("\n\n")
	}
	return b.String()
}

func collectSSE(t *testing.T, raw string) (string, error) {
	t.Helper()
	var got strings.Builder
	err := parseCompletionSSE(strings.NewReader(raw), func(s string) { got.WriteString(s) })
	return got.String(), err
}

// 正常流：文本增量应原样拼接。
func TestParseCompletionSSETextDeltas(t *testing.T) {
	raw := sseStream(
		`{"type":"content_block_start","content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"你好"}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"，世界"}}`,
		`{"type":"content_block_stop"}`,
	)
	got, err := collectSSE(t, raw)
	if err != nil {
		t.Fatalf("正常流不应报错: %v", err)
	}
	if got != "你好，世界" {
		t.Fatalf("文本增量拼接错误: %q", got)
	}
}

// 上游错误事件的 message 必须暴露给上层。
func TestParseCompletionSSEErrorEventWithMessage(t *testing.T) {
	raw := sseStream(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	_, err := collectSSE(t, raw)
	if err == nil || !strings.Contains(err.Error(), "Overloaded") {
		t.Fatalf("错误事件的 message 应被返回: %v", err)
	}
}

// 回归：上游只给 error.type（没有 message）时，以前会静默忽略该事件，
// 整轮变成"没有任何输出"，日志里也查不到原因。
func TestParseCompletionSSEErrorEventWithoutMessage(t *testing.T) {
	for _, raw := range []string{
		sseStream(`{"type":"error","error":{"type":"overloaded_error"}}`),
		sseStream(`{"type":"error"}`),
	} {
		if _, err := collectSSE(t, raw); err == nil {
			t.Fatalf("缺少 message 的错误事件也必须报错: %s", raw)
		}
	}
	_, err := collectSSE(t, sseStream(`{"type":"error","error":{"type":"rate_limit_error"}}`))
	if err == nil || !strings.Contains(err.Error(), "rate_limit_error") {
		t.Fatalf("无 message 时应回退到 error.type: %v", err)
	}
}

// 静默空回复：上游 200、没有错误事件、也没有任何文本增量。
// 解析本身不算错误（是否重试由 attemptError 判定），但会留下诊断日志。
func TestParseCompletionSSEEmptyStream(t *testing.T) {
	got, err := collectSSE(t, sseStream(`{"type":"content_block_stop"}`))
	if err != nil {
		t.Fatalf("空流不应被当成解析错误: %v", err)
	}
	if got != "" {
		t.Fatalf("空流不应产生文本: %q", got)
	}
}

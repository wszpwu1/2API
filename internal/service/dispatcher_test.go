package service

import (
	"errors"
	"fmt"
	"testing"
)

func resetModelAliases() {
	modelAliasMu.Lock()
	modelAliasCache = map[string]string{}
	modelAliasMu.Unlock()
}

func TestUpstreamModelVariant(t *testing.T) {
	cases := []struct{ in, want string }{
		{"claude-sonnet-5.5", "claude-sonnet-5-5"},
		{"claude-sonnet-5", ""},
		{"claude-sonnet-4-6", ""},
		{"claude-haiku-4-5-20251001", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := upstreamModelVariant(c.in); got != c.want {
			t.Errorf("upstreamModelVariant(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestIsUnsupportedModelErr(t *testing.T) {
	upstream := fmt.Errorf("创建会话 HTTP 400: %s",
		`{"type":"error","error":{"type":"invalid_request_error","message":"Unsupported model"}}`)
	if !isUnsupportedModelErr(upstream) {
		t.Errorf("应识别上游的 Unsupported model 错误: %v", upstream)
	}
	if isUnsupportedModelErr(nil) {
		t.Error("nil 不应被判定为模型不支持")
	}
	if isUnsupportedModelErr(errors.New("创建会话 HTTP 400: account_session_invalid")) {
		t.Error("其他错误不应被误判为模型不支持")
	}
}

func TestResolveUpstreamModelUsesCachedAlias(t *testing.T) {
	resetModelAliases()
	t.Cleanup(resetModelAliases)

	if got := resolveUpstreamModel("claude-sonnet-5.5"); got != "claude-sonnet-5.5" {
		t.Fatalf("未命中缓存时应原样返回, 得到 %q", got)
	}
	cacheModelAlias("claude-sonnet-5.5", "claude-sonnet-5-5")
	if got := resolveUpstreamModel("claude-sonnet-5.5"); got != "claude-sonnet-5-5" {
		t.Fatalf("命中缓存时应返回上游 ID, 得到 %q", got)
	}
	if got := resolveUpstreamModel("claude-sonnet-5"); got != "claude-sonnet-5" {
		t.Fatalf("无点号的模型名不应被改写, 得到 %q", got)
	}
	if got := resolveUpstreamModel(""); got != "" {
		t.Fatalf("空模型名应原样返回, 得到 %q", got)
	}
}

package service

import (
	"errors"
	"testing"
)

func TestIsUnsupportedModelErr(t *testing.T) {
	yes := []error{
		errors.New("CreateConversation: HTTP 400: {\"type\":\"invalid_request_error\",\"message\":\"Unsupported model\"}"),
		errors.New("unsupported model: claude-sonnet-5.5"),
	}
	for _, err := range yes {
		if !isUnsupportedModelErr(err) {
			t.Fatalf("%v 应判定为模型名不被上游支持", err)
		}
	}
	no := []error{
		nil,
		errors.New("HTTP 429: rate limited"),
		errors.New("account_session_invalid"),
	}
	for _, err := range no {
		if isUnsupportedModelErr(err) {
			t.Fatalf("%v 不应判定为模型名不被上游支持", err)
		}
	}
}

// 写法规范化只改点号/大小写/空白，不把 5 变成 5-5 之类的换型号行为。
func TestCanonicalModelID(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"claude-sonnet-5", "claude-sonnet-5"},
		{"claude-sonnet-5-5", "claude-sonnet-5-5"},
		{"claude-sonnet-5.5", "claude-sonnet-5-5"},
		{" Claude-Sonnet-5.5 ", "claude-sonnet-5-5"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := canonicalModelID(tc.in); got != tc.want {
			t.Fatalf("canonicalModelID(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// 缓存只影响被缓存的那个写法，未命中的模型原样透传（5 与 5-5 互不影响）。
func TestModelAliasCache(t *testing.T) {
	modelAliasMu.Lock()
	modelAliasCache = map[string]string{}
	modelAliasMu.Unlock()

	if got := resolveUpstreamModel("claude-sonnet-5-5"); got != "claude-sonnet-5-5" {
		t.Fatalf("未命中缓存时应原样返回，得到 %q", got)
	}

	cacheModelAlias("claude-sonnet-5.5", "claude-sonnet-5-5")
	if got := resolveUpstreamModel("claude-sonnet-5.5"); got != "claude-sonnet-5-5" {
		t.Fatalf("命中缓存时应返回适配后的 ID，得到 %q", got)
	}
	if got := resolveUpstreamModel("claude-sonnet-5"); got != "claude-sonnet-5" {
		t.Fatalf("其他模型不应被缓存牵连，得到 %q", got)
	}
}

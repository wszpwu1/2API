package adapter

import (
	"strings"
	"testing"

	"claude2api/internal/service"
)

func TestDetectScriptLanguage(t *testing.T) {
	tests := []struct {
		name string
		text string
		want detectedLanguage
	}{
		{"中文", "请帮我检查这段代码", languageChinese},
		{"英文", "Please review this code", languageEnglish},
		{"日文", "このコードを確認してください", languageJapanese},
		{"韩文", "이 코드를 검토해 주세요", languageKorean},
		{"中文夹英文标识符", "请修改 buildPrompt function", languageChinese},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := detectScriptLanguage(tt.text); got != tt.want {
				t.Fatalf("detectScriptLanguage(%q) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}

func TestExtractLastUserMessageIgnoresCodeAndToolResults(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "旧消息"},
		{Role: "tool", Content: "tool output in English"},
		{Role: "user", Content: "```go\nfunc main() {}\n```\n请解释问题"},
	}
	got := extractLastUserMessage(messages)
	if strings.Contains(got, "func main") || !strings.Contains(got, "请解释问题") {
		t.Fatalf("未正确过滤代码块: %q", got)
	}
}

func TestDetectLanguageAndTimezoneAutoDetect(t *testing.T) {
	prompt := service.Prompt{Text: "base"}
	lang := detectLanguageAndTimezone(&prompt, []Message{{Role: "user", Content: "日本語で説明してください"}}, clientPrefs{})
	if lang != languageJapanese {
		t.Fatalf("lang = %q, want %q", lang, languageJapanese)
	}
	if !strings.HasPrefix(prompt.AcceptLanguage, "ja-JP") {
		t.Fatalf("AcceptLanguage = %q", prompt.AcceptLanguage)
	}
	if prompt.Timezone != "Asia/Tokyo" {
		t.Fatalf("Timezone = %q", prompt.Timezone)
	}
}

func TestDetectLanguageAndTimezoneClientOverride(t *testing.T) {
	// 客户端透传语言与时区时，应覆盖自动检测结果（消息实际是中文）。
	prompt := service.Prompt{Text: "base"}
	prefs := clientPrefs{lang: "en-US,en;q=0.9", tz: "America/New_York"}
	lang := detectLanguageAndTimezone(&prompt, []Message{{Role: "user", Content: "请帮我检查代码"}}, prefs)
	if lang != languageEnglish {
		t.Fatalf("lang = %q, want %q", lang, languageEnglish)
	}
	if prompt.AcceptLanguage != "en-US,en;q=0.9" {
		t.Fatalf("AcceptLanguage = %q，客户端透传值未保留", prompt.AcceptLanguage)
	}
	if prompt.Timezone != "America/New_York" {
		t.Fatalf("Timezone = %q，客户端透传值未保留", prompt.Timezone)
	}
}

func TestDetectLanguageAndTimezoneClientLangOnly(t *testing.T) {
	// 仅透传语言时，时区应按语言推断。
	prompt := service.Prompt{Text: "base"}
	lang := detectLanguageAndTimezone(&prompt, []Message{{Role: "user", Content: "hello"}}, clientPrefs{lang: "ja"})
	if lang != languageJapanese {
		t.Fatalf("lang = %q, want %q", lang, languageJapanese)
	}
	if prompt.Timezone != "Asia/Tokyo" {
		t.Fatalf("Timezone = %q，应从语言推断", prompt.Timezone)
	}
}

func TestLangFromAcceptHeader(t *testing.T) {
	tests := []struct {
		in   string
		want detectedLanguage
	}{
		{"zh-CN,zh;q=0.9,en;q=0.8", languageChinese},
		{"zh-TW,zh;q=0.9", languageChinese},
		{"ja-JP,ja;q=0.9", languageJapanese},
		{"ko-KR,ko;q=0.9", languageKorean},
		{"en-US,en;q=0.9", languageEnglish},
		{"fr-FR,fr;q=0.9", languageUnknown},
	}
	for _, tt := range tests {
		if got := langFromAcceptHeader(tt.in); got != tt.want {
			t.Fatalf("langFromAcceptHeader(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestBuildPromptAppendsReplyInstruction(t *testing.T) {
	prompt, err := buildPrompt(
		[]Message{{Role: "user", Content: "日本語で説明してください"}},
		nil, nil, false, nil, clientPrefs{},
	)
	if err != nil {
		t.Fatalf("buildPrompt: %v", err)
	}
	if !strings.Contains(prompt.Text, "日本語で回答") {
		t.Fatalf("缺少动态语言指令: %q", prompt.Text)
	}
}

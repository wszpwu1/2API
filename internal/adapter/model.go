package adapter

import "strings"

// isSonnet5Family 判断模型是否属于 Sonnet 5 系列。
// claude-sonnet-5 与 claude-sonnet-5-5 是两个独立模型，但工具协议使用同一套
// 中性措辞；后缀 -thinking 也自然命中。这里仅用于选择工具提示词，不改变模型 ID。
func isSonnet5Family(model string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(model)), "sonnet-5")
}

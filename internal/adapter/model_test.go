package adapter

import "testing"

func TestSupportedModelsIncludesSonnet55UpstreamID(t *testing.T) {
	for _, model := range supportedModels {
		if model == "claude-sonnet-5-5" {
			return
		}
	}
	t.Fatal("supportedModels 不包含 claude-sonnet-5-5")
}
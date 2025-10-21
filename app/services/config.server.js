/**
 * Configuration Service
 * Centralizes all configuration values for the chat service
 */

export const AppConfig = {
  // API Configuration
  api: {
    defaultModel: "gemini-2.5-flash",
    maxTokens: 2000,
    defaultPromptType: "standardAssistant",
  },

  // Error Message Templates
  errorMessages: {
    missingMessage: "Message is required",
    apiUnsupported:
      "This endpoint only supports server-sent events (SSE) requests or history requests.",
    authFailed: "Authentication failed with Gemini API",
    apiKeyError: "Please check your API key in environment variables",
    rateLimitExceeded: "Rate limit exceeded",
    rateLimitDetails: "Please try again later",
    genericError: "Failed to get response from Gemini",
  },

  // Tool Configuration
  tools: {
    productSearchName: "search_shop_catalog",
    maxProductsToDisplay: 3,
  },
};

export default AppConfig;

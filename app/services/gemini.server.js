/**
 * Gemini Service
 * Manages interactions with the Google Gemini API
 */
import { GoogleGenerativeAI, HarmBlockThreshold, HarmCategory } from "@google/generative-ai";
import AppConfig from "./config.server";
import systemPrompts from "../prompts/prompts.json";

/**
 * Creates a Gemini service instance
 * @param {string} apiKey - Gemini API key
 * @returns {Object} Gemini service with methods for interacting with Gemini API
 */
export function createGeminiService(apiKey = process.env.GEMINI_API_KEY) {
  // Initialize Gemini client
  const genAI = new GoogleGenerativeAI(apiKey);

  /**
   * Streams a conversation with Gemini
   * @param {Object} params - Stream parameters
   * @param {Array} params.messages - Conversation history [{ role: 'user' | 'model' | 'system', content: '...' }]
   * @param {string} params.promptType - Type of system prompt to use
   * @param {Array} params.tools - Optional tool definitions
   * @param {Object} streamHandlers - Stream event handlers
   * @param {Function} streamHandlers.onText - Handles text chunks
   * @param {Function} streamHandlers.onMessage - Handles complete messages
   * @returns {Promise<Object>} Final message
   */
  const streamConversation = async (
    { messages, promptType = AppConfig.api.defaultPromptType, tools },
    streamHandlers
  ) => {
    const model = genAI.getGenerativeModel({
      model: AppConfig.api.defaultModel || "gemini-1.5-pro",
      tools: tools && tools.length > 0 ? tools : undefined,
      systemInstruction: getSystemPrompt(promptType),
    });

    // Prepare input by flattening messages
    const promptParts = messages.map(msg => ({
      role: msg.role === "assistant" ? "model" : msg.role,
      parts: [{ text: msg.content }],
    }));

    // Start streaming
    const result = await model.generateContentStream({
      contents: promptParts,
      generationConfig: {
        maxOutputTokens: AppConfig.api.maxTokens,
      },
      safetySettings: [
        { category: HarmCategory.HARM_CATEGORY_DEROGATORY, threshold: HarmBlockThreshold.BLOCK_NONE },
      ],
    });

    let fullText = "";

    // Listen to stream chunks
    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        fullText += text;
        if (streamHandlers.onText) streamHandlers.onText(text);
      }
    }

    // Final complete message
    if (streamHandlers.onMessage) {
      streamHandlers.onMessage(fullText);
    }

    return {
      role: "model",
      content: fullText,
    };
  };

  /**
   * Gets the system prompt for a given type
   * @param {string} promptType - The prompt type to retrieve
   * @returns {string} System prompt text
   */
  const getSystemPrompt = (promptType) => {
    return (
      systemPrompts.systemPrompts[promptType]?.content ||
      systemPrompts.systemPrompts[AppConfig.api.defaultPromptType].content
    );
  };

  return {
    streamConversation,
    getSystemPrompt,
  };
}

export default {
  createGeminiService,
};

/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.anthropic;

import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.type.TypeReference;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.anthropic.api.AnthropicApi.AnthropicMessage;
import org.springframework.ai.anthropic.api.AnthropicApi.ChatCompletionRequest;
import org.springframework.ai.anthropic.api.AnthropicApi.ChatCompletionResponse;
import org.springframework.ai.anthropic.api.AnthropicApi.ContentBlock;
import org.springframework.ai.anthropic.api.AnthropicApi.ContentBlock.Source;
import org.springframework.ai.anthropic.api.AnthropicApi.ContentBlock.Type;
import org.springframework.ai.anthropic.api.AnthropicApi.Role;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.ai.chat.observation.ChatModelObservationDocumentation;
import org.springframework.ai.chat.observation.DefaultChatModelObservationConvention;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.tool.DefaultToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.ai.support.UsageCalculator;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.util.json.JsonParser;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;

/**
 * The {@link ChatModel} implementation for the Anthropic service.
 *
 * @author Christian Tzolov
 * @author luocongqiu
 * @author Mariusz Bernacki
 * @author Thomas Vitale
 * @author Claudio Silva Junior
 * @author Alexandros Pappas
 * @author Jonghoon Park
 * @author Soby Chacko
 * @since 1.0.0
 */
public class AnthropicChatModel implements ChatModel {

	public static final String DEFAULT_MODEL_NAME = AnthropicApi.ChatModel.CLAUDE_3_7_SONNET.getValue();

	public static final Integer DEFAULT_MAX_TOKENS = 500;

	public static final Double DEFAULT_TEMPERATURE = 0.8;

	private static final Logger logger = LoggerFactory.getLogger(AnthropicChatModel.class);

	private static final ChatModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultChatModelObservationConvention();

	private static final ToolCallingManager DEFAULT_TOOL_CALLING_MANAGER = ToolCallingManager.builder().build();

	/**
	 * The retry template used to retry the OpenAI API calls.
	 */
	public final RetryTemplate retryTemplate;

	/**
	 * The lower-level API for the Anthropic service.
	 */
	private final AnthropicApi anthropicApi;

	/**
	 * The default options used for the chat completion requests.
	 */
	private final AnthropicChatOptions defaultOptions;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	private final ToolCallingManager toolCallingManager;

	/**
	 * The tool execution eligibility predicate used to determine if a tool can be
	 * executed.
	 */
	private final ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate;

	/**
	 * Conventions to use for generating observations.
	 */
	private ChatModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public AnthropicChatModel(AnthropicApi anthropicApi, AnthropicChatOptions defaultOptions,
			ToolCallingManager toolCallingManager, RetryTemplate retryTemplate,
			ObservationRegistry observationRegistry) {
		this(anthropicApi, defaultOptions, toolCallingManager, retryTemplate, observationRegistry,
				new DefaultToolExecutionEligibilityPredicate());
	}

	public AnthropicChatModel(AnthropicApi anthropicApi, AnthropicChatOptions defaultOptions,
			ToolCallingManager toolCallingManager, RetryTemplate retryTemplate, ObservationRegistry observationRegistry,
			ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {

		Assert.notNull(anthropicApi, "anthropicApi cannot be null");
		Assert.notNull(defaultOptions, "defaultOptions cannot be null");
		Assert.notNull(toolCallingManager, "toolCallingManager cannot be null");
		Assert.notNull(retryTemplate, "retryTemplate cannot be null");
		Assert.notNull(observationRegistry, "observationRegistry cannot be null");
		Assert.notNull(toolExecutionEligibilityPredicate, "toolExecutionEligibilityPredicate cannot be null");

		this.anthropicApi = anthropicApi;
		this.defaultOptions = defaultOptions;
		this.toolCallingManager = toolCallingManager;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
		this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		// Before moving any further, build the final request Prompt,
		// merging runtime and default options.
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalCall(requestPrompt, null);
	}

	public ChatResponse internalCall(Prompt prompt, ChatResponse previousChatResponse) {
		ChatCompletionRequest request = createRequest(prompt, false);

		ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
			.prompt(prompt)
			.provider(AnthropicApi.PROVIDER_NAME)
			.build();

		ChatResponse response = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {

				ResponseEntity<ChatCompletionResponse> completionEntity = this.retryTemplate.execute(
						ctx -> this.anthropicApi.chatCompletionEntity(request, this.getAdditionalHttpHeaders(prompt)));

				AnthropicApi.ChatCompletionResponse completionResponse = completionEntity.getBody();
				AnthropicApi.Usage usage = completionResponse.usage();

				Usage currentChatResponseUsage = usage != null ? this.getDefaultUsage(completionResponse.usage())
						: new EmptyUsage();
				Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentChatResponseUsage,
						previousChatResponse);

				ChatResponse chatResponse = toChatResponse(completionEntity.getBody(), accumulatedUsage);
				observationContext.setResponse(chatResponse);

				return chatResponse;
			});

		if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
			var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
			if (toolExecutionResult.returnDirect()) {
				// Return tool execution result directly to the client.
				return ChatResponse.builder()
					.from(response)
					.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
					.build();
			}
			else {
				// Send the tool execution result back to the model.
				return this.internalCall(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()),
						response);
			}
		}

		return response;
	}

	private DefaultUsage getDefaultUsage(AnthropicApi.Usage usage) {
		return new DefaultUsage(usage.inputTokens(), usage.outputTokens(), usage.inputTokens() + usage.outputTokens(),
				usage);
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		// Before moving any further, build the final request Prompt,
		// merging runtime and default options.
		Prompt requestPrompt = buildRequestPrompt(prompt);
		return this.internalStream(requestPrompt, null);
	}

	public Flux<ChatResponse> internalStream(Prompt prompt, ChatResponse previousChatResponse) {
		return Flux.deferContextual(contextView -> {
			ChatCompletionRequest request = createRequest(prompt, true);

			ChatModelObservationContext observationContext = ChatModelObservationContext.builder()
				.prompt(prompt)
				.provider(AnthropicApi.PROVIDER_NAME)
				.build();

			Observation observation = ChatModelObservationDocumentation.CHAT_MODEL_OPERATION.observation(
					this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry);

			observation.parentObservation(contextView.getOrDefault(ObservationThreadLocalAccessor.KEY, null)).start();

			Flux<ChatCompletionResponse> response = this.anthropicApi.chatCompletionStream(request,
					this.getAdditionalHttpHeaders(prompt));

			// @formatter:off
			Flux<ChatResponse> chatResponseFlux = response.flatMap(chatCompletionResponse -> {
				AnthropicApi.Usage usage = chatCompletionResponse.usage();
				Usage currentChatResponseUsage = usage != null ? this.getDefaultUsage(chatCompletionResponse.usage()) : new EmptyUsage();
				Usage accumulatedUsage = UsageCalculator.getCumulativeUsage(currentChatResponseUsage, previousChatResponse);
				ChatResponse chatResponse = toChatResponse(chatCompletionResponse, accumulatedUsage);

				if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), chatResponse)) {

					if (chatResponse.hasFinishReasons(Set.of("tool_use"))) {
						// FIXME: bounded elastic needs to be used since tool calling
						//  is currently only synchronous
						return Flux.defer(() -> {
							var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, chatResponse);
							if (toolExecutionResult.returnDirect()) {
								// Return tool execution result directly to the client.
								return Flux.just(ChatResponse.builder().from(chatResponse)
									.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
									.build());
							}
							else {
								// Send the tool execution result back to the model.
								return this.internalStream(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()),
										chatResponse);
							}
						}).subscribeOn(Schedulers.boundedElastic());
					} else {						
						return Mono.empty();
					}

				} else {
					// If internal tool execution is not required, just return the chat response.
					return Mono.just(chatResponse);
				}
			})
			.doOnError(observation::error)
			.doFinally(s -> observation.stop())
			.contextWrite(ctx -> ctx.put(ObservationThreadLocalAccessor.KEY, observation));
			// @formatter:on

			return new MessageAggregator().aggregate(chatResponseFlux, observationContext::setResponse);
		});
	}

	private ChatResponse toChatResponse(ChatCompletionResponse chatCompletion, Usage usage) {
		if (chatCompletion == null) {
			logger.warn("Null chat completion returned");
			return new ChatResponse(List.of());
		}

		try {
			ContentProcessingResult processingResult = processContentBlocks(chatCompletion);
			List<Generation> generations = createGenerations(processingResult, chatCompletion.stopReason());

			return new ChatResponse(generations, this.from(chatCompletion, usage));
		}
		catch (Exception e) {
			logger.error("Error processing chat completion response", e);
			// Return a minimal response to prevent complete failure
			return new ChatResponse(
					List.of(new Generation(new AssistantMessage("Error processing response", Map.of()),
							ChatGenerationMetadata.builder().finishReason(chatCompletion.stopReason()).build())),
					this.from(chatCompletion, usage));
		}
	}

	private ContentProcessingResult processContentBlocks(ChatCompletionResponse chatCompletion) {
		List<String> textContents = new ArrayList<>();
		List<String> webSearchTexts = new ArrayList<>();
		List<Generation> nonTextGenerations = new ArrayList<>();
		List<AssistantMessage.ToolCall> toolCalls = new ArrayList<>();

		for (ContentBlock content : chatCompletion.content()) {
			try {
				switch (content.type()) {
					case TEXT, TEXT_DELTA:
						processTextContent(content, textContents);
						break;
					case THINKING, THINKING_DELTA:
						processThinkingContent(content, nonTextGenerations, chatCompletion.stopReason());
						break;
					case REDACTED_THINKING:
						processRedactedThinkingContent(content, nonTextGenerations, chatCompletion.stopReason());
						break;
					case TOOL_USE:
						processToolUseContent(content, toolCalls);
						break;
					case SERVER_TOOL_USE:
						processServerToolContent(content, webSearchTexts);
						break;
					case WEB_SEARCH_TOOL_RESULT:
						processWebSearchContent(content, webSearchTexts);
						break;
					default:
						logger.warn("Unknown content block type: {}", content.type());
						break;
				}
			}
			catch (Exception e) {
				logger.error("Error processing content block of type: {}", content.type(), e);
				// Continue processing other blocks
			}
		}

		return new ContentProcessingResult(textContents, webSearchTexts, nonTextGenerations, toolCalls);
	}

	private void processTextContent(ContentBlock content, List<String> textContents) {
		String text = content.text();
		if (text != null && !text.trim().isEmpty()) {
			textContents.add(text);
		}
	}

	private void processThinkingContent(ContentBlock content, List<Generation> nonTextGenerations, String stopReason) {
		try {
			Map<String, Object> thinkingProperties = new HashMap<>();
			String signature = content.signature();
			if (signature != null) {
				thinkingProperties.put("signature", signature);
			}

			String thinking = content.thinking();
			nonTextGenerations.add(new Generation(new AssistantMessage(thinking, thinkingProperties),
					ChatGenerationMetadata.builder().finishReason(stopReason).build()));
		}
		catch (Exception e) {
			logger.error("Error processing thinking content", e);
		}
	}

	private void processRedactedThinkingContent(ContentBlock content, List<Generation> nonTextGenerations,
			String stopReason) {
		try {
			Map<String, Object> redactedProperties = new HashMap<>();
			Object data = content.data();
			if (data != null) {
				redactedProperties.put("data", data);
			}

			nonTextGenerations.add(new Generation(new AssistantMessage(null, redactedProperties),
					ChatGenerationMetadata.builder().finishReason(stopReason).build()));
		}
		catch (Exception e) {
			logger.error("Error processing redacted thinking content", e);
		}
	}

	private void processToolUseContent(ContentBlock content, List<AssistantMessage.ToolCall> toolCalls) {
		try {
			String functionCallId = content.id();
			String functionName = content.name();
			Object input = content.input();

			if (functionCallId != null && functionName != null && input != null) {
				String functionArguments = JsonParser.toJson(input);
				toolCalls
					.add(new AssistantMessage.ToolCall(functionCallId, "function", functionName, functionArguments));
			}
			else {
				logger.warn("Incomplete tool use content: id={}, name={}, input={}", functionCallId, functionName,
						input);
			}
		}
		catch (Exception e) {
			logger.error("Error processing tool use content", e);
		}
	}

	private void processServerToolContent(ContentBlock content, List<String> webSearchTexts) {
		try {
			Object contentObj = content.content();
			if (contentObj != null) {
				String serverToolContent = contentObj.toString();
				if (!serverToolContent.trim().isEmpty()) {
					webSearchTexts.add(serverToolContent);
					logger.debug("Processed server tool content: {} characters", serverToolContent.length());
				}
			}
		}
		catch (Exception e) {
			logger.error("Error processing server tool content", e);
		}
	}

	private void processWebSearchContent(ContentBlock content, List<String> webSearchTexts) {
		try {
			Object contentObj = content.content();
			if (contentObj != null) {
				String webSearchContent = contentObj.toString();
				if (!webSearchContent.trim().isEmpty()) {
					webSearchTexts.add(webSearchContent);
					logger.debug("Processed web search content: {} characters", webSearchContent.length());
				}
			}
		}
		catch (Exception e) {
			logger.error("Error processing web search content", e);
		}
	}

	private List<Generation> createGenerations(ContentProcessingResult result, String stopReason) {
		List<Generation> generations = new ArrayList<>();

		// Consolidate text and web search content into the first generation
		List<String> allTextContent = new ArrayList<>();
		allTextContent.addAll(result.textContents());
		allTextContent.addAll(result.webSearchTexts());

		if (!allTextContent.isEmpty()) {
			try {
				String consolidatedContent = String.join("\n", allTextContent);
				generations.add(new Generation(new AssistantMessage(consolidatedContent, Map.of()),
						ChatGenerationMetadata.builder().finishReason(stopReason).build()));

				if (!result.webSearchTexts().isEmpty()) {
					logger.debug("Consolidated {} text contents and {} web search contents",
							result.textContents().size(), result.webSearchTexts().size());
				}
			}
			catch (Exception e) {
				logger.error("Error creating consolidated content generation", e);
			}
		}

		// Add other non-text generations
		generations.addAll(result.nonTextGenerations());

		// Add empty generation if no content was processed
		if (stopReason != null && generations.isEmpty()) {
			generations.add(new Generation(new AssistantMessage(null, Map.of()),
					ChatGenerationMetadata.builder().finishReason(stopReason).build()));
		}

		// Add tool call generation if present
		if (!CollectionUtils.isEmpty(result.toolCalls())) {
			try {
				AssistantMessage assistantMessage = new AssistantMessage("", Map.of(), result.toolCalls());
				Generation toolCallGeneration = new Generation(assistantMessage,
						ChatGenerationMetadata.builder().finishReason(stopReason).build());
				generations.add(toolCallGeneration);
			}
			catch (Exception e) {
				logger.error("Error creating tool call generation", e);
			}
		}

		return generations;
	}

	private record ContentProcessingResult(List<String> textContents, List<String> webSearchTexts,
			List<Generation> nonTextGenerations, List<AssistantMessage.ToolCall> toolCalls) {
	}

	private ChatResponseMetadata from(AnthropicApi.ChatCompletionResponse result) {
		return from(result, this.getDefaultUsage(result.usage()));
	}

	private ChatResponseMetadata from(AnthropicApi.ChatCompletionResponse result, Usage usage) {
		Assert.notNull(result, "Anthropic ChatCompletionResult must not be null");
		return ChatResponseMetadata.builder()
			.id(result.id())
			.model(result.model())
			.usage(usage)
			.keyValue("stop-reason", result.stopReason())
			.keyValue("stop-sequence", result.stopSequence())
			.keyValue("type", result.type())
			.build();
	}

	private Source getSourceByMedia(Media media) {
		String data = this.fromMediaData(media.getData());

		// http is not allowed and redirect not allowed
		if (data.startsWith("https://")) {
			return new Source(data);
		}
		else {
			return new Source(media.getMimeType().toString(), data);
		}
	}

	private String fromMediaData(Object mediaData) {
		if (mediaData instanceof byte[] bytes) {
			return Base64.getEncoder().encodeToString(bytes);
		}
		else if (mediaData instanceof String text) {
			return text;
		}
		else {
			throw new IllegalArgumentException("Unsupported media data type: " + mediaData.getClass().getSimpleName());
		}

	}

	private Type getContentBlockTypeByMedia(Media media) {
		String mimeType = media.getMimeType().toString();
		if (mimeType.startsWith("image")) {
			return Type.IMAGE;
		}
		else if (mimeType.contains("pdf")) {
			return Type.DOCUMENT;
		}
		throw new IllegalArgumentException("Unsupported media type: " + mimeType
				+ ". Supported types are: images (image/*) and PDF documents (application/pdf)");
	}

	private MultiValueMap<String, String> getAdditionalHttpHeaders(Prompt prompt) {

		Map<String, String> headers = new HashMap<>(this.defaultOptions.getHttpHeaders());
		if (prompt.getOptions() != null && prompt.getOptions() instanceof AnthropicChatOptions chatOptions) {
			headers.putAll(chatOptions.getHttpHeaders());
		}
		return CollectionUtils.toMultiValueMap(
				headers.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> List.of(e.getValue()))));
	}

	Prompt buildRequestPrompt(Prompt prompt) {
		// Process runtime options
		AnthropicChatOptions runtimeOptions = null;
		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions) {
				runtimeOptions = ModelOptionsUtils.copyToTarget(toolCallingChatOptions, ToolCallingChatOptions.class,
						AnthropicChatOptions.class);
			}
			else {
				runtimeOptions = ModelOptionsUtils.copyToTarget(prompt.getOptions(), ChatOptions.class,
						AnthropicChatOptions.class);
			}
		}

		// Define request options by merging runtime options and default options
		AnthropicChatOptions requestOptions = ModelOptionsUtils.merge(runtimeOptions, this.defaultOptions,
				AnthropicChatOptions.class);

		// Merge @JsonIgnore-annotated options explicitly since they are ignored by
		// Jackson, used by ModelOptionsUtils.
		if (runtimeOptions != null) {
			if (runtimeOptions.getFrequencyPenalty() != null) {
				logger.warn("The frequencyPenalty option is not supported by Anthropic API. Ignoring.");
			}
			if (runtimeOptions.getPresencePenalty() != null) {
				logger.warn("The presencePenalty option is not supported by Anthropic API. Ignoring.");
			}
			requestOptions.setHttpHeaders(
					mergeHttpHeaders(runtimeOptions.getHttpHeaders(), this.defaultOptions.getHttpHeaders()));
			requestOptions.setInternalToolExecutionEnabled(
					ModelOptionsUtils.mergeOption(runtimeOptions.getInternalToolExecutionEnabled(),
							this.defaultOptions.getInternalToolExecutionEnabled()));
			requestOptions.setToolNames(ToolCallingChatOptions.mergeToolNames(runtimeOptions.getToolNames(),
					this.defaultOptions.getToolNames()));
			requestOptions.setToolCallbacks(ToolCallingChatOptions.mergeToolCallbacks(runtimeOptions.getToolCallbacks(),
					this.defaultOptions.getToolCallbacks()));
			requestOptions.setToolContext(ToolCallingChatOptions.mergeToolContext(runtimeOptions.getToolContext(),
					this.defaultOptions.getToolContext()));
			requestOptions.setWebSearchEnabled(ModelOptionsUtils.mergeOption(runtimeOptions.getWebSearchEnabled(),
					this.defaultOptions.getWebSearchEnabled()));
			requestOptions.setWebSearchOptions(ModelOptionsUtils.mergeOption(runtimeOptions.getWebSearchOptions(),
					this.defaultOptions.getWebSearchOptions()));
		}
		else {
			requestOptions.setHttpHeaders(this.defaultOptions.getHttpHeaders());
			requestOptions.setInternalToolExecutionEnabled(this.defaultOptions.getInternalToolExecutionEnabled());
			requestOptions.setToolNames(this.defaultOptions.getToolNames());
			requestOptions.setToolCallbacks(this.defaultOptions.getToolCallbacks());
			requestOptions.setToolContext(this.defaultOptions.getToolContext());
			requestOptions.setWebSearchEnabled(this.defaultOptions.getWebSearchEnabled());
			requestOptions.setWebSearchOptions(this.defaultOptions.getWebSearchOptions());
		}

		ToolCallingChatOptions.validateToolCallbacks(requestOptions.getToolCallbacks());

		return new Prompt(prompt.getInstructions(), requestOptions);
	}

	private Map<String, String> mergeHttpHeaders(Map<String, String> runtimeHttpHeaders,
			Map<String, String> defaultHttpHeaders) {
		var mergedHttpHeaders = new HashMap<>(defaultHttpHeaders);
		mergedHttpHeaders.putAll(runtimeHttpHeaders);
		return mergedHttpHeaders;
	}

	ChatCompletionRequest createRequest(Prompt prompt, boolean stream) {

		List<AnthropicMessage> userMessages = prompt.getInstructions()
			.stream()
			.filter(message -> message.getMessageType() != MessageType.SYSTEM)
			.map(message -> {
				if (message.getMessageType() == MessageType.USER) {
					List<ContentBlock> contents = new ArrayList<>(List.of(new ContentBlock(message.getText())));
					if (message instanceof UserMessage userMessage) {
						if (!CollectionUtils.isEmpty(userMessage.getMedia())) {
							List<ContentBlock> mediaContent = userMessage.getMedia().stream().map(media -> {
								Type contentBlockType = getContentBlockTypeByMedia(media);
								var source = getSourceByMedia(media);
								return new ContentBlock(contentBlockType, source);
							}).toList();
							contents.addAll(mediaContent);
						}
					}
					return new AnthropicMessage(contents, Role.valueOf(message.getMessageType().name()));
				}
				else if (message.getMessageType() == MessageType.ASSISTANT) {
					AssistantMessage assistantMessage = (AssistantMessage) message;
					List<ContentBlock> contentBlocks = new ArrayList<>();
					if (StringUtils.hasText(message.getText())) {
						contentBlocks.add(new ContentBlock(message.getText()));
					}
					if (!CollectionUtils.isEmpty(assistantMessage.getToolCalls())) {
						for (AssistantMessage.ToolCall toolCall : assistantMessage.getToolCalls()) {
							contentBlocks.add(new ContentBlock(Type.TOOL_USE, toolCall.id(), toolCall.name(),
									ModelOptionsUtils.jsonToMap(toolCall.arguments())));
						}
					}
					return new AnthropicMessage(contentBlocks, Role.ASSISTANT);
				}
				else if (message.getMessageType() == MessageType.TOOL) {
					List<ContentBlock> toolResponses = ((ToolResponseMessage) message).getResponses()
						.stream()
						.map(toolResponse -> new ContentBlock(Type.TOOL_RESULT, toolResponse.id(),
								toolResponse.responseData()))
						.toList();
					return new AnthropicMessage(toolResponses, Role.USER);
				}
				else {
					throw new IllegalArgumentException("Unsupported message type: " + message.getMessageType());
				}
			})
			.toList();

		String systemPrompt = prompt.getInstructions()
			.stream()
			.filter(m -> m.getMessageType() == MessageType.SYSTEM)
			.map(m -> m.getText())
			.collect(Collectors.joining(System.lineSeparator()));

		ChatCompletionRequest request = new ChatCompletionRequest(this.defaultOptions.getModel(), userMessages,
				systemPrompt, this.defaultOptions.getMaxTokens(), this.defaultOptions.getTemperature(), stream);

		AnthropicChatOptions requestOptions = (AnthropicChatOptions) prompt.getOptions();
		request = ModelOptionsUtils.merge(requestOptions, request, ChatCompletionRequest.class);

		// Add the tool definitions to the request's tools parameter.
		List<ToolDefinition> toolDefinitions = this.toolCallingManager.resolveToolDefinitions(requestOptions);
		List<AnthropicApi.ToolSpec> allTools = new ArrayList<>();

		// Add function tools
		if (!CollectionUtils.isEmpty(toolDefinitions)) {
			allTools.addAll(getFunctionTools(toolDefinitions));
		}

		// Add web search tool if enabled
		if (Boolean.TRUE.equals(requestOptions.getWebSearchEnabled())) {
			AnthropicApi.WebSearchTool webSearchTool = requestOptions.getWebSearchOptions();
			if (webSearchTool == null) {
				webSearchTool = new AnthropicApi.WebSearchTool();
			}
			allTools.add(webSearchTool);
		}

		if (!allTools.isEmpty()) {
			request = ModelOptionsUtils.merge(request, this.defaultOptions, ChatCompletionRequest.class);
			request = ChatCompletionRequest.from(request).tools(allTools).build();
		}

		return request;
	}

	private List<AnthropicApi.ToolSpec> getFunctionTools(List<ToolDefinition> toolDefinitions) {
		return toolDefinitions.stream().<AnthropicApi.ToolSpec>map(toolDefinition -> {
			var name = toolDefinition.name();
			var description = toolDefinition.description();
			String inputSchema = toolDefinition.inputSchema();
			return new AnthropicApi.Tool(name, description, JsonParser.fromJson(inputSchema, new TypeReference<>() {
			}));
		}).toList();
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return AnthropicChatOptions.fromOptions(this.defaultOptions);
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(ChatModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private AnthropicApi anthropicApi;

		private AnthropicChatOptions defaultOptions = AnthropicChatOptions.builder()
			.model(DEFAULT_MODEL_NAME)
			.maxTokens(DEFAULT_MAX_TOKENS)
			.temperature(DEFAULT_TEMPERATURE)
			.build();

		private RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;

		private ToolCallingManager toolCallingManager;

		private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

		private ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate = new DefaultToolExecutionEligibilityPredicate();

		private Builder() {
		}

		public Builder anthropicApi(AnthropicApi anthropicApi) {
			this.anthropicApi = anthropicApi;
			return this;
		}

		public Builder defaultOptions(AnthropicChatOptions defaultOptions) {
			this.defaultOptions = defaultOptions;
			return this;
		}

		public Builder retryTemplate(RetryTemplate retryTemplate) {
			this.retryTemplate = retryTemplate;
			return this;
		}

		public Builder toolCallingManager(ToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		public Builder toolExecutionEligibilityPredicate(
				ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {
			this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
			return this;
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public AnthropicChatModel build() {
			if (this.toolCallingManager != null) {
				return new AnthropicChatModel(this.anthropicApi, this.defaultOptions, this.toolCallingManager,
						this.retryTemplate, this.observationRegistry, this.toolExecutionEligibilityPredicate);
			}
			return new AnthropicChatModel(this.anthropicApi, this.defaultOptions, DEFAULT_TOOL_CALLING_MANAGER,
					this.retryTemplate, this.observationRegistry, this.toolExecutionEligibilityPredicate);
		}

	}

}

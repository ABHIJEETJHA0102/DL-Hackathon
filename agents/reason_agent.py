import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.llms import HuggingFaceHub
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)
from pydantic import Extra

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MATLABReasoningAgent")

class TroubleshootingStep(BaseModel):
    """Model for a single troubleshooting step"""
    step_number: int = Field(description="The sequence number of this troubleshooting step")
    action: str = Field(description="The specific action to take in this step")
    rationale: str = Field(description="Why this step helps diagnose or solve the problem")
    expected_outcome: str = Field(description="What should happen if this step is successful")
    contingency: Optional[str] = Field(None, description="What to do if the step doesn't produce expected results")

class RootCauseAnalysis(BaseModel):
    """Model for root cause analysis results"""
    primary_cause: str = Field(description="The most likely root cause of the issue")
    confidence: float = Field(description="Confidence level in the primary cause (0.0-1.0)")
    alternative_causes: List[str] = Field(description="Other possible causes with lower probability")
    relevant_documentation: List[str] = Field(description="References to specific MATLAB documentation sections")

class TroubleshootingPlan(BaseModel):
    """Complete troubleshooting plan with root cause and steps"""
    problem_summary: str = Field(description="Brief summary of the understood problem")
    root_cause: RootCauseAnalysis = Field(description="Analysis of probable root causes")
    steps: List[TroubleshootingStep] = Field(description="Ordered steps to diagnose and resolve the issue")
    additional_notes: Optional[str] = Field(None, description="Any extra information or caveats")

class DeepSeekChatModel(BaseChatModel, BaseLLM):
    """
    Wrapper for DeepSeek models via HuggingFace Hub API to match the ChatModel interface
    """
    repo_id: str = "deepseek-ai/deepseek-coder-33b-instruct"
    temperature: float = 0.2
    hf_model: Any = None
    
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.allow
    
    def __init__(self, repo_id: str = "deepseek-ai/deepseek-coder-33b-instruct", temperature: float = 0.2):
        """
        Initialize the DeepSeek Chat model
        
        Args:
            repo_id: The HuggingFace model repo ID
            temperature: The temperature setting for the model
        """
        super().__init__()
        self.repo_id = repo_id
        self.temperature = temperature
        
        # Initialize the underlying HuggingFaceHub model
        self.hf_model = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 2048,
                "do_sample": True,
                "top_p": 0.95
            },
            huggingfacehub_api_token="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"
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the model based on the messages"""
        # Format the conversation history into a prompt that DeepSeek can understand
        prompt = self._format_messages_to_prompt(messages)
        
        # Get response from the model
        response = self.hf_model.generate([prompt], stop=stop)
        
        # Extract the text from the LLMResult object
        response_text = response.generations[0][0].text
        
        # Return in the expected LangChain format
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )
      
    async def _agenerate(
      self,
      messages: List[BaseMessage],
      stop: Optional[List[str]] = None,
      run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
      **kwargs: Any,
  ) -> ChatResult:
      raise NotImplementedError()
    
    def _format_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a prompt string for DeepSeek models"""
        prompt_parts = []

        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"<system>\n{message.content}\n</system>")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"<user>\n{message.content}\n</user>")
            elif isinstance(message, AIMessage):
                prompt_parts.append(
                    f"<assistant>\n{message.content}\n</assistant>"
                )

        # Add the final assistant prefix to indicate we want a response
        prompt_parts.append("<assistant>")

        return "\n".join(prompt_parts)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to DeepSeek's model."""
        response = self.hf_model.generate([prompt], stop=stop)
        return response[0]


class MATLABReasoningAgent:
    """
    The Reasoning Agent that analyzes retrieved documents, identifies root causes,
    and plans step-by-step troubleshooting responses using logical reasoning.
    """
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-33b-instruct", temperature: float = 0.2):
        """
        Initialize the MATLAB Reasoning Agent
        
        Args:
            model_name: The DeepSeek model repo ID on HuggingFace Hub to use for reasoning
            temperature: The temperature setting for the LLM
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the DeepSeek model via HuggingFace
        self.llm = DeepSeekChatModel(repo_id=model_name, temperature=temperature)
        self.output_parser = PydanticOutputParser(pydantic_object=TroubleshootingPlan)
        
        # Load reasoning prompt template
        self.reasoning_prompt = self._create_reasoning_prompt()
        self.reasoning_chain = LLMChain(
            llm=self.llm,
            prompt=self.reasoning_prompt
        )
        
        logger.info(f"MATLAB Reasoning Agent initialized with DeepSeek model: {model_name}")
    
    def _create_reasoning_prompt(self) -> PromptTemplate:
        """Create the prompt template for the reasoning agent"""
        template = """
        You are an expert MATLAB troubleshooting assistant with deep knowledge of MATLAB's components,
        common errors, and best practices. Your goal is to analyze the user's problem and retrieved
        documentation to create a comprehensive troubleshooting plan.

        ## USER QUERY
        {user_query}

        ## RETRIEVED DOCUMENTATION
        {retrieved_docs}

        ## INSTRUCTIONS
        1. Carefully analyze both the user query and the retrieved documentation.
        2. Identify the most likely root cause(s) of the issue.
        3. Develop a logical, step-by-step troubleshooting plan.
        4. Ensure each step has a clear rationale and expected outcome.
        5. Include contingency steps for potential complications.
        6. Cite specific sections from the documentation in your analysis.
        7. Use your expert knowledge of MATLAB to fill any gaps not covered in the documentation.

        ## OUTPUT FORMAT
        {format_instructions}
        """
        
        return PromptTemplate(
            input_variables=["user_query", "retrieved_docs"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
            template=template
        )
    
    def reason(self, user_query: str, retrieved_docs: List[Dict[str, Any]]) -> TroubleshootingPlan:
        """
        Generate a troubleshooting plan based on the user query and retrieved documentation
        
        Args:
            user_query: The original user problem query
            retrieved_docs: List of relevant documentation chunks with metadata
        
        Returns:
            A structured troubleshooting plan
        """
        # Start timing
        start_time = datetime.now()
        
        # Log the reasoning request
        logger.info(f"Starting reasoning process for query: {user_query[:100]}...")
        
        # Format retrieved docs for prompt
        formatted_docs = self._format_retrieved_docs(retrieved_docs)
        
        enhanced_prompt = self.reasoning_prompt.format(
            user_query=user_query,
            retrieved_docs=formatted_docs
        ) + "\n\nRemember to provide a proper JSON object with problem_summary, root_cause, and steps fields, not just the schema definition."
        
        # Generate the reasoning output
        try:
            chain_output = self.llm(enhanced_prompt)
            
            # Extract JSON from the response - sometimes models add extra text
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', chain_output)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON between curly braces
                json_match = re.search(r'(\{[\s\S]*\})', chain_output)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = chain_output
                    
            # Clean up any trailing commas which can cause JSON parsing errors
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Try to parse JSON directly first
            try:
                parsed_json = json.loads(json_str)
                troubleshooting_plan = TroubleshootingPlan(**parsed_json)
            except (json.JSONDecodeError, TypeError, ValidationError) as e:
                logger.warning(f"Failed to parse direct JSON. Trying with output parser: {str(e)}")
                troubleshooting_plan = self.output_parser.parse(json_str)
            
            # Log success
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Reasoning completed successfully in {duration:.2f} seconds")
            
            return troubleshooting_plan
            
        except Exception as e:
            logger.error(f"Error during reasoning process: {str(e)}")
        
            # Fallback: Create a basic troubleshooting plan
            fallback_plan = TroubleshootingPlan(
                problem_summary=f"Failed to automatically analyze: {user_query}",
                root_cause=RootCauseAnalysis(
                    primary_cause="Parser error - unable to generate structured analysis",
                    confidence=0.1,
                    alternative_causes=["Model output format mismatch"],
                    relevant_documentation=[]
                ),
                steps=[
                    TroubleshootingStep(
                        step_number=1,
                        action="Review the error manually",
                        rationale="Automated analysis failed",
                        expected_outcome="Manual resolution",
                        contingency="Contact support"
                    )
                ],
                additional_notes=f"Error during processing: {str(e)}"
            )
            return fallback_plan
    
    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format the retrieved documentation chunks for inclusion in the prompt"""
        formatted_docs = []
        
        for i, doc in enumerate(docs, 1):
            # Extract key information from the document
            content = doc.get("content", "")
            title = doc.get("metadata", {}).get("title", f"Document {i}")
            source = doc.get("metadata", {}).get("source", "Unknown")
            
            # Format the document
            formatted_doc = f"--- DOCUMENT {i}: {title} (Source: {source}) ---\n{content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def analyze_error_patterns(self, error_message: str) -> Dict[str, Any]:
        """
        Analyze error message patterns to identify common MATLAB error types
        
        Args:
            error_message: The error message from MATLAB
            
        Returns:
            Dictionary with error analysis
        """
        # Common MATLAB error patterns
        error_patterns = {
            "undefined_function": r"Undefined function or variable '([^']+)'",
            "index_exceeds": r"Index exceeds (matrix|array) dimensions",
            "type_mismatch": r"Error using .+ \nOperands to the .+ must be",
            "file_not_found": r"File '([^']+)' not found",
            "syntax_error": r"Error: ([^\n]+)\n"
        }
        
        # Analyze the error message
        results = {
            "original_error": error_message,
            "detected_patterns": []
        }
        
        for error_type, pattern in error_patterns.items():
            matches = re.findall(pattern, error_message)
            if matches:
                results["detected_patterns"].append({
                    "type": error_type,
                    "matches": matches
                })
        
        return results
    
    def evaluate_confidence(self, user_query: str, troubleshooting_plan: TroubleshootingPlan) -> float:
        """
        Evaluate confidence in the generated troubleshooting plan
        
        Args:
            user_query: The original user problem query
            troubleshooting_plan: The generated plan
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Implement confidence evaluation logic
        # This could be based on:
        # - How specifically the plan addresses the user query
        # - Number of relevant documentation sources
        # - Specificity of the steps
        # - Coverage of potential alternative causes
        
        # Simple placeholder implementation
        confidence_score = min(troubleshooting_plan.root_cause.confidence, 0.95)
        
        if len(troubleshooting_plan.steps) < 2:
            confidence_score *= 0.7  # Penalize very short plans
            
        if not troubleshooting_plan.root_cause.relevant_documentation:
            confidence_score *= 0.8  # Penalize plans without documentation references
            
        return confidence_score

# Example usage
if __name__ == "__main__":
    # Make sure HuggingFace API key is set
    # if "HF_API_KEY" not in os.environ:
    #     print("ERROR: Please set the HF_API_KEY environment variable.")
    #     print("You can get an API key from https://huggingface.co/settings/tokens")
    #     exit(1)
    
    # Initialize the reasoning agent with DeepSeek model
    agent = MATLABReasoningAgent(
        model_name="deepseek-ai/deepseek-coder-33b-instruct",  # Use DeepSeek coding model
        temperature=0.2
    )
    
    # Example user query
    user_query = "I'm getting 'Index exceeds matrix dimensions' error when trying to access array elements in a for loop"
    
    # Example retrieved documents (would normally come from Knowledge Retriever Agent)
    retrieved_docs = [
        {
            "content": "The error 'Index exceeds matrix dimensions' occurs when you try to access an element outside the defined size of an array. For example, if A is a 3x3 matrix, attempting to access A(4,4) will trigger this error.",
            "metadata": {
                "title": "Common MATLAB Errors",
                "source": "MATLAB Documentation: Array Indexing"
            }
        },
        {
            "content": "When using arrays in for loops, ensure your loop indices do not exceed array bounds. A common mistake is using 'for i = 1:length(A)' for both dimensions of a 2D array. Instead, use 'for i = 1:size(A,1)' and 'for j = 1:size(A,2)'.",
            "metadata": {
                "title": "For Loop Best Practices",
                "source": "MATLAB Documentation: Control Flow"
            }
        }
    ]
    
    # Generate a troubleshooting plan
    plan = agent.reason(user_query, retrieved_docs)
    
    # Print the plan
    print(json.dumps(plan.dict(), indent=2))
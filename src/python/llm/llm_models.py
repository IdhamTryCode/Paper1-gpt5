#!/usr/bin/env python3
"""
Comprehensive LLM Model Implementations

This module implements all specified LLM models with advanced features
for benchmarking between Python and Rust implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
    AutoModelForTokenClassification, AutoModelForMaskedLM,
    GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer,
    DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer,
    AlbertModel, AlbertTokenizer, GenerationConfig
)
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GPT2Model:
    """GPT-2 model with comprehensive features."""
    
    def __init__(self, model_name: str = "gpt2", use_half_precision: bool = False):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply optimizations
        if use_half_precision and self.device.type == 'cuda':
            self.model = self.model.half()
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded GPT-2 model: {model_name}")
    
    def generate_text(self, prompt: str, max_length: int = 50, 
                     temperature: float = 0.7, top_k: int = 50, 
                     top_p: float = 0.9, repetition_penalty: float = 1.0) -> List[str]:
        """Generate text with comprehensive parameters."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        generation_config = GenerationConfig(
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        return torch.exp(loss).item()


class BERTModel:
    """BERT model with comprehensive features."""
    
    def __init__(self, model_name: str = "bert-base-uncased", task: str = "classification"):
        self.model_name = model_name
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer based on task
        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        elif task == "masked_language_modeling":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            self.model = BertModel.from_pretrained(model_name)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded BERT model: {model_name} for task: {task}")
    
    def classify_text(self, texts: List[str], return_probs: bool = True) -> List[Dict[str, Any]]:
        """Classify text using the loaded model."""
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
                
                result = {
                    "text": text,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
                
                if return_probs:
                    result["probabilities"] = probabilities.cpu().numpy().tolist()
                
                results.append(result)
        
        return results
    
    def question_answering(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Perform question answering using the loaded model."""
        results = []
        
        for question, context in zip(questions, contexts):
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                
                answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                start_confidence = torch.softmax(outputs.start_logits, dim=-1).max().item()
                end_confidence = torch.softmax(outputs.end_logits, dim=-1).max().item()
                confidence = (start_confidence + end_confidence) / 2
                
                result = {
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "confidence": confidence,
                    "start_position": answer_start.item(),
                    "end_position": answer_end.item()
                }
                
                results.append(result)
        
        return results


class DistilBERTModel:
    """DistilBERT model with comprehensive features."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", task: str = "classification"):
        self.model_name = model_name
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer based on task
        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            self.model = DistilBertModel.from_pretrained(model_name)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded DistilBERT model: {model_name} for task: {task}")
    
    def classify_text(self, texts: List[str], return_probs: bool = True) -> List[Dict[str, Any]]:
        """Classify text using the loaded model."""
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
                
                result = {
                    "text": text,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
                
                if return_probs:
                    result["probabilities"] = probabilities.cpu().numpy().tolist()
                
                results.append(result)
        
        return results


class RoBERTaModel:
    """RoBERTa model with comprehensive features."""
    
    def __init__(self, model_name: str = "roberta-base", task: str = "classification"):
        self.model_name = model_name
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer based on task
        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            self.model = RobertaModel.from_pretrained(model_name)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded RoBERTa model: {model_name} for task: {task}")
    
    def classify_text(self, texts: List[str], return_probs: bool = True) -> List[Dict[str, Any]]:
        """Classify text using the loaded model."""
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
                
                result = {
                    "text": text,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
                
                if return_probs:
                    result["probabilities"] = probabilities.cpu().numpy().tolist()
                
                results.append(result)
        
        return results


class ALBERTModel:
    """ALBERT model with comprehensive features."""
    
    def __init__(self, model_name: str = "albert-base-v2", task: str = "classification"):
        self.model_name = model_name
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer based on task
        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            self.model = AlbertModel.from_pretrained(model_name)
        
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded ALBERT model: {model_name} for task: {task}")
    
    def classify_text(self, texts: List[str], return_probs: bool = True) -> List[Dict[str, Any]]:
        """Classify text using the loaded model."""
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1).values.item()
                
                result = {
                    "text": text,
                    "predicted_class": predicted_class,
                    "confidence": confidence
                }
                
                if return_probs:
                    result["probabilities"] = probabilities.cpu().numpy().tolist()
                
                results.append(result)
        
        return results


def create_llm_model(model_name: str, task: str = "text-generation", **kwargs) -> Any:
    """Factory function to create LLM models."""
    
    model_configs = {
        "gpt2": {
            "class": GPT2Model,
            "default_params": {"model_name": "gpt2", "use_half_precision": False}
        },
        "gpt2-medium": {
            "class": GPT2Model,
            "default_params": {"model_name": "gpt2-medium", "use_half_precision": False}
        },
        "gpt2-large": {
            "class": GPT2Model,
            "default_params": {"model_name": "gpt2-large", "use_half_precision": False}
        },
        "bert-base-uncased": {
            "class": BERTModel,
            "default_params": {"model_name": "bert-base-uncased", "task": task}
        },
        "bert-large-uncased": {
            "class": BERTModel,
            "default_params": {"model_name": "bert-large-uncased", "task": task}
        },
        "distilbert-base-uncased": {
            "class": DistilBERTModel,
            "default_params": {"model_name": "distilbert-base-uncased", "task": task}
        },
        "roberta-base": {
            "class": RoBERTaModel,
            "default_params": {"model_name": "roberta-base", "task": task}
        },
        "albert-base-v2": {
            "class": ALBERTModel,
            "default_params": {"model_name": "albert-base-v2", "task": task}
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = model_configs[model_name]
    model_class = config["class"]
    default_params = config["default_params"].copy()
    
    # Update with provided parameters
    default_params.update(kwargs)
    
    return model_class(**default_params)


def get_model_info(model: Any) -> Dict[str, Any]:
    """Get comprehensive information about a model."""
    
    if hasattr(model, 'model'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    if hasattr(model, 'model'):
        for param in model.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
    else:
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "model_name": getattr(model, 'model_name', 'unknown'),
        "architecture": model.__class__.__name__
    } 
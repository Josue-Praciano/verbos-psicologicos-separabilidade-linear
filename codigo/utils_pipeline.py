import torch
import numpy as np
from transformers import AutoTokenizer, XLMRobertaModel
import re

def inicializar_modelos(model_name='xlm-roberta-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device

def limpar_e_filtrar_texto(texto):
    """Aplica regras de limpeza mantendo a estrutura de espaços intacta."""
    if not texto:
        return ""
    # Remove espaços excessivos nas pontas e colapsa múltiplos espaços em um só
    texto_limpo = texto.strip()
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
    return texto_limpo

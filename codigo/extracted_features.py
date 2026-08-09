import json
import numpy as np
import torch  
from pathlib import Path
from utils_pipeline import inicializar_modelos

def extrair_embedding_mean_pooling(batch_itens, tokenizer, model, device):
    """
    Extrai embeddings usando Mean Pooling global sobre toda a sentença.
    Faz a média de todos os tokens válidos (não-padding), utilizando a 
    média das últimas 4 camadas do XLM-R. Suporta batches.
    """
    textos = [item['texto'] for item in batch_itens]

    with torch.no_grad():
        # Tokeniza o lote de textos com padding e truncamento dinâmico
        encoded = tokenizer(
            textos, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        attention_mask = encoded['attention_mask']  # Shape: [batch_size, seq_len]
        outputs = model(**encoded, output_hidden_states=True)
        
        # Média aritmética das últimas 4 camadas do Transformer
        ultimas_4 = torch.stack(outputs.hidden_states[-4:])
        media_camadas = ultimas_4.mean(dim=0)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Expande a attention_mask para o tamanho do embedding [batch_size, seq_len, hidden_size]
        # Isso garante que multiplicaremos por 0 os tokens de padding ([PAD])
        mask_expandida = attention_mask.unsqueeze(-1).expand(media_camadas.size()).float()
        
        # Soma os embeddings das posições válidas (onde a máscara de atenção é 1)
        soma_embeddings = torch.sum(media_camadas * mask_expandida, dim=1)
        
        # Calcula a quantidade de tokens reais (não-padding) para cada sentença do batch
        soma_mask = torch.clamp(mask_expandida.sum(dim=1), min=1e-9)
        
        # Divide para obter o Mean Pooling (média global da sentença)
        embeddings_finais = (soma_embeddings / soma_mask).cpu().numpy()
        
    return embeddings_finais

def main():
    base_path = Path(__file__).parent.parent
    json_path = base_path / 'data' / 'corpus_processado.json'
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dados_corpus = json.load(f)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {json_path}")
        return
    
    print(f"Total de registros carregados: {len(dados_corpus)}")
    
    tokenizer, model, device = inicializar_modelos('xlm-roberta-base')
    
    features_list = []
    
    # Configuração de Batching
    BATCH_SIZE = 16  # 16 funcionou bem para GPU de 4GB
    
    print(f"Iniciando a extração com Mean Pooling (Batch Size: {BATCH_SIZE})...")
    
    for i in range(0, len(dados_corpus), BATCH_SIZE):
        batch_itens = dados_corpus[i:i + BATCH_SIZE]
        
        # Passa o batch completo de dicionários para extrair a média global das sentenças
        embeddings_batch = extrair_embedding_mean_pooling(batch_itens, tokenizer, model, device)
        
        features_list.append(embeddings_batch)
        
        if (i + len(batch_itens)) % 100 == 0 or (i + len(batch_itens)) == len(dados_corpus):
            print(f"Processados {i + len(batch_itens)}/{len(dados_corpus)} itens...")

    # Consolida e exporta os embeddings extraídos
    if features_list:
        features_matrix = np.vstack(features_list)
        print("\nProcessamento concluído com sucesso!")
        print("Formato final da matriz de features (Mean Pooling + 4 camadas):", features_matrix.shape)
        
        # 1. Define o caminho da pasta outputs a partir da raiz do projeto
        pasta_outputs = base_path / 'outputs'
        
        # 2. Garante que a pasta 'outputs' exista (cria se necessário)
        pasta_outputs.mkdir(parents=True, exist_ok=True)
        
        # 3. Altera o nome do arquivo de saída para refletir a nova estratégia
        caminho_salvamento = pasta_outputs / 'features_xlmr_mean_pooling.npy'
        
        # 4. Salva a matriz no local correto
        np.save(caminho_salvamento, features_matrix)
        print(f"Embeddings salvos com sucesso em: {caminho_salvamento}")
    else:
        print("Nenhuma feature foi extraída.")

if __name__ == "__main__":
    main()
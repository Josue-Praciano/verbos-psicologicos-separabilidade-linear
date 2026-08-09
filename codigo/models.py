import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, silhouette_score

# Dicionário de mapeamento conforme a sua definição de classes
MAPEAMENTO_CLASSES = {
    # Classe 1
    "admirar": "Classe 1", "adorar": "Classe 1", "amar": "Classe 1", "desejar": "Classe 1", 
    "detestar": "Classe 1", "estranhar": "Classe 1", "invejar": "Classe 1", "odiar": "Classe 1", 
    "almejar": "Classe 1", "apreciar": "Classe 1",
    # Classe 2
    "aborrecer": "Classe 2", "afligir": "Classe 2", "abalar": "Classe 2", "chatear": "Classe 2", 
    "comover": "Classe 2", "decepcionar": "Classe 2", "encantar": "Classe 2", "escandalizar": "Classe 2", 
    "apaixonar": "Classe 2", "constranger": "Classe 2",
    # Classe 3
    "acalmar": "Classe 3", "apaziguar": "Classe 3", "convencer": "Classe 3", "enganar": "Classe 3", 
    "humilhar": "Classe 3", "ludibriar": "Classe 3", "pacificar": "Classe 3", "derrotar": "Classe 3", 
    "ridicularizar": "Classe 3", "tranquilizar": "Classe 3",
    # Classe 4
    "acovardar": "Classe 4", "agradar": "Classe 4", "alarmar": "Classe 4", "alegrar": "Classe 4", 
    "animar": "Classe 4", "apavorar": "Classe 4", "assombrar": "Classe 4", "assustar": "Classe 4", 
    "aterrorizar": "Classe 4", "atormentar": "Classe 4",
    # Classe Controle
    "dar": "Classe controle", "atribuir": "Classe controle", "devolver": "Classe controle", "distribuir": "Classe controle", 
    "emprestar": "Classe controle", "entregar": "Classe controle", "fornecer": "Classe controle", "oferecer": "Classe controle", 
    "pagar": "Classe controle", "transferir": "Classe controle"
}

def carregar_dados_e_labels(json_path, features_path):
    json_path = Path(json_path)
    features_path = Path(features_path)

    X = np.load(features_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        
    assert len(dados_json) == X.shape[0], "O número de linhas do JSON difere da matriz de features!"
    
    # Extrai os verbos originais e remove espaços/capitalização
    verbos_originais = [item['verbo_alvo'].strip().lower() for item in dados_json]
    
    # Mapeia cada verbo para sua respectiva classe (1 a 4)
    classes_texto = []
    for v in verbos_originais:
        classe = MAPEAMENTO_CLASSES.get(v)
        if not classe:
            raise ValueError(f"O verbo '{v}' encontrado no JSON não está mapeado em nenhuma classe no script!")
        classes_texto.append(classe)
    
    # Encoder para as classes (y representará as Classes de 1 a 4)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(classes_texto)
    
    # Retornamos também a lista de verbos originais correspondentes a cada linha para rastreamento
    return X, y, np.array(verbos_originais), label_encoder

def calcular_separabilidade(X, y):
    score_geral = silhouette_score(X, y, metric='cosine')
    print("\n" + "="*50)
    print("Separabilidade")
    print("="*50)
    print(f"Silhouette Score Geral (Métrica de Cosseno): {score_geral:.4f}")
    print("="*50 + "\n")

def gerar_relatorio_por_verbo(verbos_test, y_test, y_pred, label_encoder):
    """
    Rastreia o desempenho individual de cada verbo no conjunto de teste.
    """
    verbos_unicos = np.unique(verbos_test)
    linhas_relatorio = []
    
    linhas_relatorio.append("\n" + "-"*65)
    linhas_relatorio.append(f"{'VERBO':<18} | {'CLASSE REAL':<12} | {'ACERTOS / TOTAL':<15} | {'ACURÁCIA':<10}")
    linhas_relatorio.append("-"*65)
    
    for verbo in sorted(verbos_unicos):
        indices = np.where(verbos_test == verbo)[0]
        
        y_test_verbo = y_test[indices]
        y_pred_verbo = y_pred[indices]
        
        acertos = np.sum(y_test_verbo == y_pred_verbo)
        total = len(indices)
        acuracia_verbo = acertos / total if total > 0 else 0
        
        classe_real_str = label_encoder.inverse_transform([y_test_verbo[0]])[0]
        
        linhas_relatorio.append(
            f"{verbo:<18} | {classe_real_str:<12} | {acertos:>6}/{total:<8} | {acuracia_verbo:>8.1%}"
        )
        
    linhas_relatorio.append("-"*65 + "\n")
    return "\n".join(linhas_relatorio)

def avaliar_e_salvar_modelo(modelo, X_train, X_test, y_train, y_test, verbos_test, label_encoder, nome_modelo, pasta_outputs):
    print(f"\n" + "="*50)
    print(f" Avaliando modelo: {nome_modelo}")
    print("="*50)
    
    # VALIDAÇÃO CRUZADA (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=26)
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')
    
    texto_cv = (
        f"Resultados da Validação Cruzada (5 Folds):\n"
        f"Acurácias por fold: {', '.join([f'{s:.2%}' for s in scores_cv])}\n"
        f"Média do CV: {scores_cv.mean():.2%} (+/- {scores_cv.std():.2%})\n"
    )
    print(texto_cv)
    
    # Treinamento e Predição no Teste
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc_teste = accuracy_score(y_test, y_pred)
    relatorio_texto = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    
    print(f"Resultados no Conjunto de Teste Final")
    print(f"Acurácia no Teste: {acc_teste:.4%}")
    print(relatorio_texto)
    
    # Gerando o relatório detalhado por Verbo
    relatorio_verbos = gerar_relatorio_por_verbo(verbos_test, y_test, y_pred, label_encoder)
    print("Desempenho detalhado por Verbo no teste:")
    print(relatorio_verbos)
    
    # Salvando Outputs
    nome_limpo = nome_modelo.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(modelo, pasta_outputs / f"modelo_{nome_limpo}.joblib")
    
    with open(pasta_outputs / f"resultado_{nome_limpo}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Resultados do {nome_modelo}\n")
        f.write(f"{texto_cv}\n")
        f.write(f"Acurácia no Teste Final: {acc_teste:.4%}\n\n{relatorio_texto}\n")
        f.write("="*65 + "\n")
        f.write("Desempenho por verbo no conjunto de teste:\n")
        f.write(relatorio_verbos)

def main():
    pasta_src = Path(__file__).resolve().parent
    raiz_projeto = pasta_src.parent
    pasta_outputs = raiz_projeto / 'outputs'
    pasta_outputs.mkdir(parents=True, exist_ok=True)
    
    json_path = raiz_projeto / 'data' / 'corpus_processado.json'
    features_path = pasta_outputs / 'features_xlmr_mean_pooling.npy'
    
    # Agora carrega X, y, os verbos individuais e o codificador de classes
    X, y, verbos, label_encoder = carregar_dados_e_labels(json_path, features_path)
    
    # Salva o label encoder atualizado (agora contendo as 4 classes como labels)
    joblib.dump(label_encoder, pasta_outputs / 'label_encoder.joblib')
    
    calcular_separabilidade(X, y)
    
    # Divisão mantendo a correspondência dos verbos individuais
    # Aumentamos o max_iter da Regressão Logística para prevenir o erro de convergência
    X_train, X_test, y_train, y_test, _, verbos_test = train_test_split(
        X, y, verbos, test_size=0.2, random_state=26, stratify=y
    )
    
    modelo_svm = SVC(kernel='linear', C=1.0, random_state=26)
    modelo_lr = LogisticRegression(max_iter=3000, random_state=26) # Aumentado max_iter para 3000
    
    avaliar_e_salvar_modelo(modelo_svm, X_train, X_test, y_train, y_test, verbos_test, label_encoder, "SVM Linear", pasta_outputs)
    avaliar_e_salvar_modelo(modelo_lr, X_train, X_test, y_train, y_test, verbos_test, label_encoder, "Regressao Logistica", pasta_outputs)

if __name__ == "__main__":
    main()
from pathlib import Path
from prov.model import ProvDocument

def generate_complete_pipeline_provenance(
    corpus_puro_path: str = "data/corpus_puro.txt",  
    preprocessing_script_path: str = "data/import_json.py",  
    corpus_path: str = r"Projeto_Fundamentos_DS\data\corpus_processado.json",
    embeddings_path: str = "outputs/features_xlmr_mean_pooling.npy",
    tsne_plot_path: str = "outputs/visualizacao_embeddings_tsne_controle.png", 
    model_name: str = "xlm-roberta-base",
    layer_used: str = "mean_last_4_layers",
    pooling_strategy: str = "masked_mean_pooling",
    classifier_name: str = "Support Vector Machine (SVC)",
    classifier_name_2: str = "Logistic Regression",
    researcher_name: str = "Josue David Praciano",
    output_prov_path: str = "código/pipeline_provenance.json"
) -> str:

    # Define a pasta de outputs dinamicamente com base no tsne_plot_path fornecido
    pasta_outputs = Path(tsne_plot_path).parent

    doc = ProvDocument()

    # Definição dos namespaces
    doc.add_namespace('prov', 'http://www.w3.org/ns/prov#')
    doc.add_namespace('ln', 'http://example.org/linguistica_psicologica#')

    # AGENTES
    ag_researcher = doc.agent('ln:researcher', {'prov:type': 'prov:Person', 'ln:name': researcher_name})
    ag_pipeline_script = doc.agent('ln:pipeline_script', {'prov:type': 'prov:SoftwareAgent'})
    
    ag_import_script = doc.agent('ln:import_json_script', {
        'prov:type': 'prov:SoftwareAgent',
        'prov:location': preprocessing_script_path,
        'ln:description': 'Script responsável por converter o corpus bruto em JSON estruturado'
    })
    
    doc.actedOnBehalfOf(ag_pipeline_script, ag_researcher)
    doc.actedOnBehalfOf(ag_import_script, ag_researcher)

   
    # Etapa 1: Extração Manual
    ent_site_origem = doc.entity('ln:corpus_do_portugues_site', {
        'prov:type': 'prov:Collection',
        'ln:url': 'https://www.corpusdoportugues.org/',
        'ln:description': 'Dados brutos extraídos manualmente do site Corpus do Português'
    })
    
    act_coleta_manual = doc.activity('ln:web_scraping_and_extraction')
    doc.wasAssociatedWith(act_coleta_manual, ag_researcher)
    doc.used(act_coleta_manual, ent_site_origem)

    # O arquivo gerado pela extração manual
    ent_corpus_puro = doc.entity('ln:corpus_puro_bruto', {
        'prov:type': 'ln:RawCorpus',
        'prov:location': corpus_puro_path,
        'ln:description': 'Dados originais salvos localmente em formato de texto antes da estruturação'
    })
    doc.wasGeneratedBy(ent_corpus_puro, act_coleta_manual)
    doc.wasDerivedFrom(ent_corpus_puro, ent_site_origem)

    act_preprocessing = doc.activity('ln:json_structuring_and_preprocessing', other_attributes={
        'ln:script_executed': 'import_json.py'
    })
    doc.wasAssociatedWith(act_preprocessing, ag_import_script)
    doc.used(act_preprocessing, ent_corpus_puro)

    # Geração do arquivo JSON estruturado contendo as chaves que você descreveu
    ent_corpus = doc.entity('ln:processed_json_corpus', {
        'prov:type': 'ln:CorpusData',
        'prov:location': str(corpus_path),
        'ln:format': 'json',
        'ln:extracted_fields': 'id, posicao_do_verbo, texto, classe, verbo_alvo'
    })
    doc.wasGeneratedBy(ent_corpus, act_preprocessing)
    doc.wasDerivedFrom(ent_corpus, ent_corpus_puro)


    # Etapa 3: Extração de Embeddings (XLM-R-base Mean Pooling .npy)
    ent_model = doc.entity('ln:transformer_model', {
        'ln:model_id': model_name, 
        'ln:layers_aggregated': layer_used,
        'ln:pooling_technique': pooling_strategy,
        'ln:masking_applied': 'True (attention_mask used to ignore padding tokens)'
    })

    act_extraction = doc.activity('ln:tokenization_and_extraction')
    doc.wasAssociatedWith(act_extraction, ag_pipeline_script)
    doc.used(act_extraction, ent_corpus)
    doc.used(act_extraction, ent_model)

    # A matriz de features final no formato numpy
    ent_embeddings = doc.entity('ln:embeddings_matrix', {
        'prov:type': 'ln:FeatureMatrix', 
        'prov:location': embeddings_path,
        'ln:format': 'npy',
        'ln:embedding_aggregation': 'sentence_level_via_mean_pooling_over_last_4_layers'
    })
    doc.wasGeneratedBy(ent_embeddings, act_extraction)
    doc.wasDerivedFrom(ent_embeddings, ent_corpus)
    doc.wasDerivedFrom(ent_embeddings, ent_model)

    # Etapa 4: CLASSIFICAÇÃO, SEPARABILIDADE E VISUALIZAÇÃO COM T-SNE (MATPLOTLIB)
  
    act_analysis = doc.activity('ln:model_training_evaluation_and_visualization')
    doc.wasAssociatedWith(act_analysis, ag_pipeline_script)
    doc.used(act_analysis, ent_embeddings)

    # Saída: Label Encoder
    ent_label_encoder = doc.entity('ln:label_encoder_file', {
        'prov:type': 'ln:DeploymentArtifact',
        'prov:location': str(pasta_outputs / 'label_encoder.joblib'),
        'ln:purpose': 'Mapeamento de categorias de texto para índices numéricos'
    })
    doc.wasGeneratedBy(ent_label_encoder, act_analysis)

    # Saída: Gráfico estático do t-SNE usando Matplotlib
    ent_tsne_plot = doc.entity('ln:tsne_scatterplot', {
        'prov:type': 'prov:Image',
        'prov:location': str(tsne_plot_path),
        'ln:dimension_reduction': 't-SNE',
        'ln:visualization_library': 'matplotlib',
        'ln:metric': 'Silhouette Score Geral (Cosseno)'
    })
    doc.wasGeneratedBy(ent_tsne_plot, act_analysis)
    doc.wasDerivedFrom(ent_tsne_plot, ent_embeddings)

    # Modelos serializados gerados na Etapa 2
    ent_modelo_svm_bin = doc.entity('ln:serialized_svm_model', {
        'prov:type': 'ln:SerializedModel',
        'prov:location': str(pasta_outputs / 'modelo_svm_linear.joblib'),
        'ln:algorithm': classifier_name
    })
    doc.wasGeneratedBy(ent_modelo_svm_bin, act_analysis)

    # Modelos de Regressão Logística
    ent_modelo_lr_bin = doc.entity('ln:serialized_lr_model', {
        'prov:type': 'ln:SerializedModel',
        'prov:location': str(pasta_outputs / 'modelo_regressao_logistica.joblib'),
        'ln:algorithm': classifier_name_2
    })
    doc.wasGeneratedBy(ent_modelo_lr_bin, act_analysis)

    # Relatórios textuais de métricas
    ent_svm_report = doc.entity('ln:svm_results_report', {
        'prov:type': 'ln:EvaluationReport',
        'prov:location': str(pasta_outputs / 'resultados_Support_Vector_Machine.txt'),
        'ln:classifier_used': classifier_name,
        'ln:kernel': 'linear'
    })
    doc.wasGeneratedBy(ent_svm_report, act_analysis)
    doc.wasDerivedFrom(ent_svm_report, ent_embeddings)

    ent_lr_report = doc.entity('ln:logistic_regression_report', {
        'prov:type': 'ln:EvaluationReport',
        'prov:location': str(pasta_outputs / 'resultados_Logistic_Regression.txt'),
        'ln:classifier_used': classifier_name_2
    })
    doc.wasGeneratedBy(ent_lr_report, act_analysis)
    doc.wasDerivedFrom(ent_lr_report, ent_embeddings)

    # Etapa 5: VISUALIZAÇÃO INTERATIVA COM PLOTLY
    
    act_visualization = doc.activity('ln:hyperplane_pca_projection_and_visualization')
    doc.wasAssociatedWith(act_visualization, ag_pipeline_script)

    # Consumo dos arquivos gerados nas etapas anteriores
    doc.used(act_visualization, ent_label_encoder)
    doc.used(act_visualization, ent_modelo_svm_bin)
    doc.used(act_visualization, ent_modelo_lr_bin)

    # Geração dos plots interativos finais
    ent_plot_lr = doc.entity('ln:plotly_lr_hyperplanes', {
        'prov:type': 'prov:Image',
        'ln:visualization_tool': 'Plotly Express',
        'ln:source_coefficients': 'Logistic Regression (.coef_ OVR)',
        'ln:target_groups': 'Classe 1, Classe 2, Classe 3, Classe 4 e Classe Controle'
    })
    doc.wasGeneratedBy(ent_plot_lr, act_visualization)
    doc.wasDerivedFrom(ent_plot_lr, ent_modelo_lr_bin)
    doc.wasDerivedFrom(ent_plot_lr, ent_embeddings)

    ent_plot_svm = doc.entity('ln:plotly_svm_hyperplanes', {
        'prov:type': 'prov:Image',
        'ln:visualization_tool': 'Plotly Express',
        'ln:source_coefficients': 'SVM Linear (Hiperplanos OVO Reconstruídos cumulativamente)',
        'ln:target_groups': 'Classe 1, Classe 2, Classe 3, Classe 4 e Classe Controle'
    })
    doc.wasGeneratedBy(ent_plot_svm, act_visualization)
    doc.wasDerivedFrom(ent_plot_svm, ent_modelo_svm_bin)
    doc.wasDerivedFrom(ent_plot_svm, ent_embeddings)

    # SERIALIZAR E SALVAR O ARQUIVO 
    json_data = doc.serialize(format='json', indent=2, ensure_ascii=False)

    caminho_final = Path(output_prov_path)
    caminho_final.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_final, 'w', encoding='utf-8') as f:
        f.write(json_data)

    print(f"[PROV] Grafo gerado com a linhagem histórica completa guardado em: {caminho_final.resolve()}")
    return json_data

if __name__ == "__main__":
    generate_complete_pipeline_provenance()

import ollama
import pandas as pd
from gensim.models import LdaModel
from gensim import corpora
import time

###############################

def basic_chat(prompt, modelo):
    
    resposta = ollama.chat(
        model=modelo,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    #print(resposta["message"]["content"])
    
    return (resposta["message"]["content"])

modelo = "gemma3n:e4b" # "gemma3:12b" ou "gemma3:27b" ou "llama3.2-vision" ou "gpt-oss:latest" ou gemma3:4b ou gemma3n:e4b

base_prompt = '''
Você é um sistema especializado em analisar as informações referentes a um tópico, obtido através de LDA, e identificar sobre qual assunto o tópico aborda.
As informações que você deve analisar são as seguintes:
Os termos mais relevantes para o tópico analisado são: {} estando eles em ordem de relevância.
Os documentos identificados pelo sistema como os mais relevantes para o tópico em questão são os seguintes:
{}
Com base nas informações fornecidas, qual é o assunto tratado pelo tópico, em síntese, e qual é a opinião geral em relação ao tema tratado?
'''

###############################

###############################

def llm_analysis (dataframe, lda_top_n_docs_p_topic, lda_model, framing, partido):
    print(".....................................")
    print("... Função llm_analysis iniciada! ...")
    
    NUMERO_TOPICOS = lda_model.num_topics
    
    print ("... Analisando discursos do partido: "+partido+" sob o recorte: "+framing+" com "+str(NUMERO_TOPICOS)+" tópicos encontrados ...")
    
    analises = []
    
    for topico_id in range(NUMERO_TOPICOS):
        print ("...")
        print ("... Partido: "+partido)
        print ("... "+framing)
        print ("... Análise dp tópico "+str(topico_id))        

        top_terms = lda_model.show_topic(topico_id, topn = 35)
        string_termos = ""
        for termo, peso in top_terms:
            string_termos = string_termos + "'" + termo + "', "
            #print(f"{termo} ({peso:.4f})")
        #print(string_termos)
        
        tmp_df = lda_top_n_docs_p_topic[lda_top_n_docs_p_topic["topic"] == topico_id].sort_values(by="probability", ascending=False).head(4).copy()
        
        string_docs = ''
        tmp_count = 1
        for doc in tmp_df['raw_disc']:
            #print(doc)
            string_docs = string_docs + "Documento " + str(tmp_count) + ": \"" + doc + "\";\n "
            tmp_count += 1
        #print(string_docs)
        
        prompt = base_prompt.format(string_termos,string_docs)
        print("... Prompt usado ...")
        print(prompt)
        print(">>>")



        main_start_time = time.time()

        resposta = basic_chat(prompt, modelo)
        
        print("... Análise gerada ...")
        print(">>>\n"+resposta+"\n<<<")
                
        analises.append(resposta)
        
        with open(f"running_files/analysis/analysis_partido {partido}_framing {framing}_topico {topico_id}.txt", "w") as text_file:
            text_file.write(prompt)
            text_file.write("\n>>> >>> >>>\n\n")
            text_file.write(resposta)

        print (">>> TEMPO: %s segundos para a geração de análise pelo SLM <<<" % (time.time() - main_start_time))
        print("......................")
        
        
        
    print("... Função llm_analysis encerrada! ...")
    print("......................................")
    
    return analises
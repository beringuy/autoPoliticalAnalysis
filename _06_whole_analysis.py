
import ollama
import os
import pandas as pd
import time

###############################

df = pd.read_csv ("running_files/political_discourses.csv")

files = os.listdir("running_files/analysis")

framing = ["before", "after"]
partidos = df["partido"].value_counts().index.tolist()
partidos

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

modelo = "gemma3:12b" # "gemma3:12b" ou "gemma3:27b"

base_prompt = '''
Você é um sistema especializado em fazer a síntese e encontrar aspectos relevantes de diferentes análises de discursos políticos realizados em plenário.
As análises realizadas foram as seguintes:
{}
Com base nas análises fornecidas, quais são os temas mais recorrentes para o partido em questão?
Qual é o grau de relevância de cada um desses temas para o partido em questão?
Qual é a visão geral dos parlamentares do partido em relação ao tema? Como se manifesta essa visão?
'''

###############################


for tmp_partido in partidos:
    for current_framing in framing:
        print("=== === === === === === === === === === === === === === ===")
        print ("Partido:",tmp_partido,"Recorte:",current_framing)
        selected_files = [a for a in files if f"{tmp_partido}_framing {current_framing}" in a]
        print(selected_files)
        
        texto = ""
        contador = 1
        for selected_file in selected_files:
            with open("running_files/analysis/"+selected_file, "r", encoding="utf-8") as f:
                conteudo = f.read()
                
                prompt, analise = conteudo.split(">>> >>> >>>", 1)
                #prompt = prompt.strip()
                analise = analise.strip()
                
                texto += "Análise {}:\n".format(contador) + analise + "\n\n"
            contador += 1
        
        prompt = base_prompt.format(texto)
        print(prompt)

        main_start_time = time.time()
        resposta = basic_chat(prompt, modelo)
        print(resposta)
        print (">>> TEMPO: %s segundos para a geração de análise pelo SLM <<<" % (time.time() - main_start_time))
        print("=== === === === === === === === === === === === === === ===")
        
        with open("running_files/final_analysis/{}.txt".format(tmp_partido+"_"+current_framing), "w") as text_file:
            text_file.write(resposta)
        
        #break
    #break

###############################


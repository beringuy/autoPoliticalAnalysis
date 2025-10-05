
from _01_list_extract import list_extract
from _02_discourses_extract import discourses_extract
from _03_preprocessing import preprocessing
from _04_topics import topics_main
from _05_llm_analysis import llm_analysis

import pandas as pd
from gensim.models import LdaModel

import os
import shutil

############## ############## ############## ############## ##############

############## ############## ############## ############## ##############

# * >>> unpack a tupla que é retornada
#df, name = discourses_extract ( *list_extract("09/07/2025", "10/07/2025") )
'''
for i in df["raw_disc"]:
    print (i)
    #print (i[0:500])
    print ("=============")


import re
re.sub(r"(?<=[^À-Ÿà-ÿ\w\.])[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z] \w*", lambda m: m.group(0).replace(" ", ""), "os homens e m ulheres que, com dedicação. O profissional liberal é aquele que atua de forma autônoma, sem vínculo empregatício. Senhor Presidente, Senhoras e Senhores Deputados, é com satisfação que subo a essa T ribuna para participar d esta Sessão em homenagem")
#'''

############## ############## ############## ############## ##############

############## ############## ############## ############## ##############

#df_before, name_before = discourses_extract ( *list_extract("09/05/2025", "08/07/2025") )
#df_after, name_after = discourses_extract ( *list_extract("09/07/2025", "09/09/2025") )

name_before = "backup/discursos02RAW_2794 discursos_ini 09052025_fim 08072025_consulta 20250919 1204.csv"
name_after = "backup/discursos02RAW_3033 discursos_ini 09072025_fim 09092025_consulta 20250919 1211.csv"

df_before = pd.read_csv(name_before)
df_after = pd.read_csv(name_after)

selected_parties = ['PL',
                    'PT',
                    'NOVO',
                    'PSOL',
                    'REPUBLICANOS',
                    'UNIÃO',
                    'MDB',
                    'PSD',
                    'PP',
                    'PDT',
                    'PODE',] # + de 50 discursos antes e depois da data chave

df_before = df_before[df_before['partido'].isin(selected_parties)]
df_after = df_after[df_after['partido'].isin(selected_parties)]

### ### ###

#df_before, name_before = preprocessing ( df_before , name_before )
#df_after, name_after   = preprocessing (  df_after , name_after  )

name_before_postpre = "backup/discursos03PREPROCESS_2794 discursos_ini 09052025_fim 08072025_consulta 20250919 1204.csv"
name_after_postpre = "backup/discursos03PREPROCESS_3033 discursos_ini 09072025_fim 09092025_consulta 20250919 1211.csv"
df_before = pd.read_csv(name_before_postpre)
df_after = pd.read_csv(name_after_postpre)

############## ############## ############## ############## ##############

############## ############## ############## ############## ##############

#df_before.columns
#df_after.iloc[2549]["link_discurso"]
#df_after.iloc[2549]["raw_disc"]
#'''
for i in df_before["preprocess_disc"]:
    print (i)
    #print (i[0:500])
    print ("=============")

for i in df_after["preprocess_disc"]:
    print (i)
    #print (i[0:500])
    print ("=============")
#'''
############## ############## ############## ############## ##############

############## ############## ############## ############## ##############
#'''

##############

if os.path.exists("running_files/lda_files/before"):
    shutil.rmtree("running_files/lda_files/before")
else:
    pass

if os.path.exists("running_files/lda_files/after"):
    shutil.rmtree("running_files/lda_files/after")
else:
    pass

### ### ###

os.makedirs("running_files/lda_files/before", exist_ok=True)
os.makedirs("running_files/lda_files/after", exist_ok=True)

##############

# LDA - BEFORE:
for partido in selected_parties:
    print ("... Modelando tópicos dos discursos do partido: " + partido + " ...")    
    topics_main( df_before[df_before["partido"] == partido] , partido )

for partido in selected_parties:
    try:
        shutil.move("running_files/lda_files/"+partido , "running_files/lda_files/before/"+partido)
    except:
        pass

##############

# LDA - AFTER:
for partido in selected_parties:
    print ("... Modelando tópicos dos discursos do partido: " + partido + " ...")    
    topics_main( df_after[df_after["partido"] == partido] , partido )

for partido in selected_parties:
    try:
        shutil.move("running_files/lda_files/"+partido , "running_files/lda_files/after/"+partido)
    except:
        pass
#'''
############## ############## ############## ############## ##############

############## ############## ############## ############## ##############

before_after = ["before", "after"]

for framing in before_after:    
    for partido in selected_parties:
        tmp_folder = "running_files/lda_files/"+framing+"/"+partido+"/"
        print(tmp_folder)
        
        topNdocd_df = pd.read_csv(tmp_folder+"lda_topN_docs_por_topico.csv")
        
        if framing == "before":
            tmp_df = df_before[df_before["partido"] == partido].copy()
            topNdocd_df = topNdocd_df.merge(df_before[['preprocess_disc' , 'raw_disc']])
        else:
            tmp_df = df_after[df_after["partido"] == partido].copy()
            topNdocd_df = topNdocd_df.merge(df_after[['preprocess_disc' , 'raw_disc']])
        
        #print(tmp_df)
        print("---------------")
        #print(topNdocd_df)
        print("===============\n")

        llm_analysis (tmp_df, # dataframe
                      #tmp_folder+"lda_distribuicao_docs.csv",
                      topNdocd_df, # docs mais relevantes por tópico
                      LdaModel.load(tmp_folder+"lda_model.model"), # modelo LDA
                      #tmp_folder+"lda_dictionary.dict",
                      framing, # recorte Before ou After
                      partido, # partido
                      )

############## ############## ############## ############## ##############

############## ############## ############## ############## ##############


'''
tmp_file = "discursos02RAW_1841 discursos_ini 09062025_fim 09072025_consulta 20250916 1643.csv"

tmp_df = pd.read_csv("backup/" + tmp_file)

tmp_df
tmp_df.columns
tmp_df['partido'].value_counts()
tmp_df['partido'].value_counts().sum()
len(tmp_df['partido'].value_counts())
tmp_df['data']

tmp_df[tmp_df['partido'] == 'PL']

tmp_df[tmp_df.isna().any(axis=1)]

###############

from gensim.models import LdaModel

tmp_lda_model = LdaModel.load("running_files/lda_files/after/MDB/lda_model.model")

tmp_lda_model.num_topics



import pandas as pd
tmp_test = pd.read_csv("running_files/lda_files/after/MDB/lda_topN_docs_por_topico.csv").merge(df_after[['preprocess_disc' , 'raw_disc']])

tmp_test.sort_values("probability", ascending=False)

tmp_test.reset_index(drop=True)
'''
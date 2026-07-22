import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import transformers
import torch
from os import listdir
from pathlib import Path
from numpy.linalg import norm
import requests

useOllama = True
#model_id = "google/gemma-3-12b-it"
#model_id = "qwen3.5:27b"
model_id = "gemma4:e4b"
#model_id = "gemma4:26b"

torch.set_float32_matmul_precision('high')

qconfig = BitsAndBytesConfig(load_in_8bit=True,llm_int8_enable_fp32_cpu_offload=True)
dtype = torch.bfloat16

if not useOllama:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=qconfig,
        torch_dtype="auto",)

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
else:
    tokenizer = None

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True, torch_dtype=torch.bfloat16)

def getAnswer(user_input,tokenizer,chat_history, debug=False):
    chat_history = chat_history + [{"role": "user", "content": user_input}]
    if not useOllama:
        prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=8192)
        fullChat = tokenizer.decode(outputs[0])
        print("Context Length: "+str(len(fullChat)))
        if(debug):
            print(fullChat)
            print('-----------------------')
        answer = fullChat[fullChat.rfind("<start_of_turn>model")+21:len(fullChat)-14]
    else:
        url = "http://localhost:11438/api/chat"
        data = {
            "model": model_id,
            "messages": chat_history,
            "options": {
                "temperature": 0,
                "top_k": 0,
                "top_p": 0,
                "num_ctx": 16392
            },
            "stream": False
        }
        resp = requests.post(url, json=data)
        if "message" in resp.json():
            answer = resp.json()["message"]["content"]
        else:
            print(resp.json())
    chat_history = chat_history + [{"role": "model", "content": answer}]
    return chat_history, answer

def filterContaining(wList, file_content):
    found = []
    li = wList.split('<li>')
    del li[0]
    for w in li:
        wArr = w.split('</li>')
        word = wArr[0].strip()
        if '(' in word:
            word = word[:word.index('(')-1]
        if word.lower() in file_content.lower():
            found.append(word)
    ret = '<ol name="Liste der Tätigkeiten">\n'
    for x in found:
        ret += '<li>'+x+'</li>\n'
    ret+='</ol>'
    return ret

def addExamples(kategorien, exListe):
    mapping = {}
    li = exListe.split('<td>')
    if len(li)==1:
        return kategorien
    del li[0]
    if len(li)%2==1:
        return kategorien
    for i in range(0,len(li),2):
        wArr = li[i].split('</td>')
        katArr = li[i+1].split('</td>')
        kat = katArr[0].strip()
        if kat in mapping:
            mapping[kat].append(wArr[0].strip())
        else:
            mapping[kat] = [wArr[0].strip()]
    ret = ''
    li = kategorien.split('</tr>')
    ret+=li[0]+' <th>Beispiele</th>\n  </tr>'
    del li[0]
    katArr = kategorien.split('<td>')
    del katArr[0]
    for i in range(len(li)-1):
        x = li[i]
        ret+=x+' <td>'
        kat = katArr[i*3].split('</td>')[0].strip()
        if kat in mapping:
            for y in mapping[kat]:
                ret+=y+', '
            ret = ret[:-2]
        ret+='</td>\n  </tr>'
    ret+=li[-1]
    return ret

path = './crawlTexte'
Path('./out').mkdir(exist_ok=True)

#for file in listdir(path):
jobs = ['job15.txt','job20.txt','job9.txt','job23.txt','job39.txt','job51.txt','job53.txt','job68.txt','job177.txt','job268.txt','job331.txt','job356.txt','job378.txt','job398.txt','job557.txt','job1067.txt','job2719.txt']
for file in jobs:
    with open(path+'/'+file, 'r', encoding="cp1252") as f:
        file_content = f.read()
        
        chat_history = [{"role": "system", "content": 'Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein, die näher mit Adjektiven beschrieben werden. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind (z.B. "Gegenstand: Pflanzen; Tätigkeit: herstellen").'}]
        inputMsg='Gegeben sei die folgende Berufsbeschreibung:\n'+ file_content +' Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind. Deine Aufgabe ist es, aus dem Text eine Liste mit allen im Text vorkommenden Tätigkeiten zu extrahieren, z.B. analysieren, Installation. Überlege zunächst, welche Tätigkeiten in der Berufsbeschreibung explizit angesprochen werden. Überprüfe dann, ob du wirklich alle Tätigkeiten erfasst hast. Bitte nutze für die Liste folgende HTML-Formatvorlage: \n html<ol name="Liste (Beispiel)"> <li> z.B. beobachtet </li> <li>z.B. anwenden </li> <li>z.B. Konzeption</li>'
        chat_history, answer = getAnswer(inputMsg,tokenizer,chat_history)
        #tätigkeitenListe = filterContaining(answer, file_content)
        tätigkeitenListe = answer
        print(answer)
        with open('./out1_sys/li_'+file+'.html', "w") as out_file:
            out_file.write(answer)
        msg2 = 'Bearbeite die zuvor erstellte Liste mit Tätigkeiten weiter, indem die Tätigkeiten zu einem Kategoriensystem verdichtest. Wähle für den Katgeoriennamen ausschließlich Verben. Bitte nutze für das Kategoriensystem folgende HTML-Formatvorlage: \n html<table name="Kategoriensystem (Beispiel)"> <thead> <tr> <th>Name</th> <th>Beschreibung</th> <th>Abgrenzung zu anderen Kategorien</th> </tr> </thead> <tbody><tr> <td>z.B. "Verkaufen"</td> <td>z.B. "Den Gegenstand gegen Bezahlung einer anderen Person als Eigentum überlassen, Interesse anderer Personen am Gegenstand forcieren "</td> <td>z.B. "Vs. Erklären: Beschaffenheit des Gegenstands muss nicht unbedingt wahrheitsgetreu übermittelt werden </td></tr> \n\n Bitte fülle dann die vorgegebene Formatvorlage für diese Tätigkeiten aus. Bilde die Kategorien so, dass ihnen eine oder mehrere Tätigkeiten aus der Liste zugeordnet werden könnten und das alle Tätigkeit aus der Liste in genau einer Kategorie wiederzufinden wäre. Die Zuordnung selbst ist nicht deine Aufgabe, sondern wird im Anschluss von einem anderen Mitarbeiter übernommen.'
        chat_history, answer = getAnswer(msg2,tokenizer,chat_history)
        answer = answer.replace('<b>','')
        answer = answer.replace('</b>','')
        kategorien = answer
        print(answer)
        with open('./out1_sys/tab_'+file+'.html', "w") as out_file:
            out_file.write(answer)
        chat_history = [{"role": "system", "content": 'Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein, die näher mit Adjektiven beschrieben werden. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind (z.B. "Gegenstand: Pflanzen; Tätigkeit: herstellen").'}]
        msg3 = 'Gegeben sei die folgende Berufsbeschreibung:\n'+ file_content +'\n\n Ordne alle Elemente aus folgender Liste: \n '+tätigkeitenListe+' \n\n den folgenden Kategorien zu: \n '+kategorien+' \n\n Jedes Element muss genau einer Kategorie zugeordnet werden. Orientiere dich an der Beschreibung der Kategorien und der Berufsbeschreibung als Kontext. Verwende die folgende HTML-Formatvorlage: \n html<table name="Zuordnung (Beispiel)"> <thead> <tr> <th>Element der Liste</th> <th>Kategorie</th> </tr> </thead> <tbody><tr> <td>z.B. "Vermarkten"</td> <td>z.B. "Verkaufen"</td></tr>'
        chat_history, answer = getAnswer(msg3,tokenizer,chat_history)
        print(answer)
        with open('./out1_sys/zuord_'+file+'.html', "w") as out_file:
            out_file.write(answer)
        final = addExamples(kategorien, answer)
        # Idee: Hier nur kat welche auch ein bsp haben
        print(final)
        with open('./out1_sys/final_'+file+'.html', "w") as out_file:
            out_file.write(final)

def readLabelAndBeschreibung(fileContend, mapping):
    li = fileContend.split('<td>')
    del li[0]
    for i in range(0,len(li),4):
        wArr = li[i].split('</td>')
        kat = wArr[0].strip()
        bArr = li[i+1].split('</td>')
        besch = bArr[0].strip()
        if kat in mapping:
            mapping[kat].append(besch)
        else:
            mapping[kat] = [besch]
    return mapping

def stripwhite(text):
    lst = text.replace('\n', ' ').split('"')
    for i, item in enumerate(lst):
        if not i % 2:
            lst[i] = item.replace(' ', '')
    return '"'.join(lst)

def readJsonList(answer):
    answer = stripwhite(answer)
    li = answer.split('["')
    if len(li)<2:
        return []
    li = li[1].split('"]')
    li = li[0].split('","')
    return li

def readJsonLabelAndDef(answer):
    li = answer.split('"Label": "')
    zw = li[1].split('"')
    name = zw[0]
    li = answer.split('"Definition": "')
    zw = li[1].split('"')
    definition = zw[0]
    return (name, definition)

mapping = {}

for file in jobs:
    with open('./out1_sys/'+'final_'+file+'.html', 'r', encoding="utf-8") as f:
        file_content = f.read()
        mapping = readLabelAndBeschreibung(file_content,mapping)
print(mapping)

##############################

def readJsonBeschreibung(answer):
    li = answer.split('"Beschreibung": "')
    zw = li[1].split('"')
    besch = zw[0]
    return besch

for kat, beschArr in mapping.items():
    if len(beschArr)>1:
        li = beschArr
        beschr = ''
        for item in li:
            beschr+= item+'\n'
        chat_history = [{"role": "system", "content": 'Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein, die näher mit Adjektiven beschrieben werden. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind (z.B. "Gegenstand: Pflanzen; Tätigkeit: herstellen").'}]
        inputMsg=('Gegeben seien folgende Beschreibungen:\n\n'+
            beschr+
            '\n\nDeine Aufgabe ist es, diese Beschreibungen zu einer einzigen im selben Stil zusammenzufassen. Die Antwort soll so knapp wie möglich sein.'+
            ' Antworte mit einem JSON-Objekt {"Beschreibung": "..."}.'
         )
        chat_history, answer = getAnswer(inputMsg,tokenizer,chat_history)
        print(answer)
        mapping[kat] = [readJsonBeschreibung(answer)]
print(mapping)

##############################

with open('./out2_sys/sys_Original.txt', "w") as out_file:
    txtFile = ''
    for kat, beschArr in mapping.items():
        txtFile+=kat+': '+beschArr[0]+'\n'
    out_file.write(txtFile)

epoch = 0
while len(mapping)>20:
    index = 0
    mappingNew = {}
    while len(mapping) > 1: #index < len(mapping):
        li = []
        for kat, beschArr in mapping.items():
            li.append(kat+': '+beschArr[0])
        embeddings = embedding_model.encode(li)
        coses = {}
        for i in range(0,len(li)):
            if i!=0:
                coses[i] = cos_sim(embeddings[0], embeddings[i])
        coses = {k: v for k, v in sorted(coses.items(), key=lambda item: item[1], reverse=True)}
        maxOKSimi = coses[list(coses.keys())[0]]-0.1
        kats = li[0]+'\n'
        for i in list([k for k, v in coses.items() if v>=maxOKSimi]):
            #print(str(coses[i])+" "+li[i])
            kats += li[i]+'\n'

        katName = list(mapping.keys())[0]
        chat_history = [{"role": "system", "content": 'Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein, die näher mit Adjektiven beschrieben werden. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind (z.B. "Gegenstand: Pflanzen; Tätigkeit: herstellen").'}]
        inputMsg=('Gegeben sei folgende Liste von Kategorien:\n\n'+
            kats+
            '\n\nDeine Aufgabe ist es, Synonyme aus der Liste zu identifizieren für die Kategorie '+katName+
            '. Gib aus der Liste die Labels der Kategorien an, die man noch unter einem gemeinsamen Begriff zusammenfassen könnte ("Liste"). Antworte mit einem JSON-Objekt {"Liste": ["..."]}'
            )
        chat_history, answer = getAnswer(inputMsg,tokenizer,chat_history)
        print(answer)
        with open('./out2_sys/json_'+str(epoch)+'_'+str(index)+'.txt', "w") as out_file:
            out_file.write(answer)
        zusammenzufassen = readJsonList(answer)
        idx_map = {key: i for i, key in enumerate(mapping)}

        numFound = 0
        kats = ''
        foundID = idx_map.get(katName)
        if foundID!=None:
            kats += li[foundID]+'\n'
            numFound+=1
        
        for k in zusammenzufassen:
            foundID = idx_map.get(k)
            if foundID!=None:
                kats += li[foundID]+'\n'
                numFound+=1
            else:
                print('ERROR: '+k+' is not a valid Kat')

        if numFound<=1:
            mappingNew[katName] = mapping[katName]
            if katName in mapping:
                del mapping[katName]
        else:
            chat_history = [{"role": "system", "content": 'Du bist Spezialist für das Entwickeln von Kategorien, für die qualitative Inhaltsanalyse. Die Kategorien sind für einen Interessentest gedacht. In den Kategorien unterscheidest du streng zwischen "Gegenstand" und "Tätigkeit". "Gegenstände" können abstrakte, konkrete oder auch ideelle Umweltausschnitte sein, die näher mit Adjektiven beschrieben werden. "Tätigkeiten" sind ausschließ Verben (auch substantiviert). Später werden die Kategorien wieder in Kombination zusammengebracht, um Berufe zu finden, in denen bestimmte Kombinationen von Tätigkeit und Gegenstand wiederzufinden sind (z.B. "Gegenstand: Pflanzen; Tätigkeit: herstellen").'}]
            inputMsg=('Gegeben seien folgende Kategorien:\n\n'+
                kats +
                '\n\n Bestimme ein gemeinsames Label, das die Bedeutung der Synonyme auf den Punkt bringt und gib eine neue Definition im Stil der gegebenen Definitionen.'+
                ' Das Label soll aus 1 bis 3 Verben in deutscher Sprache bestehen. Es darf auch ein Label aus der Liste sein, sofern es repräsentativ für alle anderen Labels ist.'+
                ' Antworte mit einem JSON-Objekt {"Label": "...", "Definition": "..."}.'
                )
            chat_history, answer = getAnswer(inputMsg,tokenizer,chat_history)
            print(answer)
            with open('./out2_sys/kat_'+str(epoch)+'_'+str(index)+'.txt', "w") as out_file:
                out_file.write(answer)
            name, definition = readJsonLabelAndDef(answer)
            for key in zusammenzufassen:
                if key in mapping:
                    del mapping[key]
            if katName in mapping:
                del mapping[katName]
            mappingNew[name] = [definition]
        
        with open('./out2_sys/sys_'+str(epoch)+'_'+str(index)+'.txt', "w") as out_file:
            txtFile = ''
            for kat, beschArr in mapping.items():
                txtFile+=kat+': '+beschArr[0]+'\n'
            out_file.write(txtFile)

        index +=1

    for key in mapping:
        mappingNew[key] = mapping[key]
    mapping = mappingNew
    with open('./out2_sys/sys_'+str(epoch)+'_end.txt', "w") as out_file:
        txtFile = ''
        for kat, beschArr in mapping.items():
            txtFile+=kat+': '+beschArr[0]+'\n'
        out_file.write(txtFile)
    epoch += 1
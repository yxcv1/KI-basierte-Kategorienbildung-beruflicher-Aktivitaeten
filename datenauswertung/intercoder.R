library("readxl")
library(DescTools)

files <- list.files("./xlsx/")
df <- data.frame()

i<-1
for (f in files) {
  if (grepl( "Segmente", f, fixed = TRUE)) {
    zw<-cbind(read_excel(paste0("./xlsx/",f), 1)[,c(1,2,7)],ks=i)
    i<-i+1
    df<-rbind(df,zw)
  }
}
df[,2]<-tolower(df[,2])
df[,2] <- gsub("untersuchen, bewerten; optimieren", "untersuchen, bewerten und optimieren", df[,2])
df[,2] <- gsub("deligieren", "delegieren", df[,2])

ksList <- list()
for (ks in 1:6) {
  tab<-read_excel(paste0("./xlsx/ks",ks,".xlsx"), 1)
  ksList[[ks]]<-list()
  id<-1
  for (i in 1:length(unlist(tab[,1]))) {
    if (is.na(unlist(tab[i,2]))) {
      print(paste0(ks," kat: ",unlist(tab[i,1])))
    }
    ksList[[ks]][[tolower(unlist(tab[i,1]))]]<-list(id=id,def=unlist(tab[i,2]))
    id<-id+1
  }
  
}

vorhandensein <- list()
anzahl<-list()
kodierungenV<-list()
kodierungenA<-list()
for (ks in 1:6) {
  vorhandensein[[ks]]<-data.frame()
  anzahl[[ks]]<-data.frame()
  kodierungenV[[ks]]<-list()
  kodierungenA[[ks]]<-list()
  for (doc in unique(df[,1])) {
    filtered <- df[df[,"ks"]==ks & df[,1]==doc,]
    rater<-unique(filtered[,3])
    if (length(rater)!=2) {
      print("Falsche Anzahl Rater")
      next
    }
    found<-list()
    found[[rater[1]]]<-rep.int(0,length(ksList[[ks]]))
    found[[rater[2]]]<-rep.int(0,length(ksList[[ks]]))
    foundAz<-list()
    foundAz[[rater[1]]]<-rep.int(0,length(ksList[[ks]]))
    foundAz[[rater[2]]]<-rep.int(0,length(ksList[[ks]]))
    for (i in 1:length(filtered[,1])) {
      row <- unlist(filtered[i,])
      row[2]<-tolower(row[2])
      if (row[2] != "sonstiges") {
        if (found[[row[3]]][ksList[[ks]][[row[2]]][["id"]]] == 0) {
          found[[row[3]]][ksList[[ks]][[row[2]]][["id"]]] <- 1
        }
        foundAz[[row[3]]][ksList[[ks]][[row[2]]][["id"]]] <- foundAz[[row[3]]][ksList[[ks]][[row[2]]][["id"]]] + 1
      }
    }
    kodierungenV[[ks]][[paste0(doc,"_r1")]]<-found[[rater[1]]]
    kodierungenV[[ks]][[paste0(doc,"_r2")]]<-found[[rater[2]]]
    kodierungenA[[ks]][[paste0(doc,"_r1")]]<-foundAz[[rater[1]]]
    kodierungenA[[ks]][[paste0(doc,"_r2")]]<-foundAz[[rater[2]]]
    for (i in 1:length(ksList[[ks]])) {
      vorhandensein[[ks]]<-rbind(vorhandensein[[ks]],c(ks,doc,found[[rater[1]]][i],found[[rater[2]]][i]))
      anzahl[[ks]]<-rbind(anzahl[[ks]],c(ks,doc,foundAz[[rater[1]]][i],foundAz[[rater[2]]][i]))
    }
  }
}

cat("Stufe 1: Vorhandensein")
for (ks in 1:6) {
  ratertab <- xtabs (~ vorhandensein[[ks]][,3] + vorhandensein[[ks]][,4])
  #print(ratertab)
  cat(paste0("Kategoriensystem: ",ks,"\n"))
  (print(CohenKappa(ratertab, conf.level = 0.95)))
  cat("\n")
}

cat("Stufe 2: Häufigkeit")
for (ks in 1:6) {
  allLevels<-union(anzahl[[ks]][,3],anzahl[[ks]][,4])
  v1<-factor(anzahl[[ks]][,3], levels = allLevels)
  v2<-factor(anzahl[[ks]][,4], levels = allLevels)
  ratertab <- xtabs (~ v1 + v2)
  cat(paste0("Kategoriensystem: ",ks,"\n"))
  (print(CohenKappa(ratertab, conf.level = 0.95)))
  cat("\n")
}

for (ks in 1:6) {
  len <- length(kodierungenV[[ks]])
  li <- kodierungenV[[ks]]
  cat(paste0("Kategoriensystem: ",ks,"\n"))
  for (i in 1:len){
    #for (i in 1:50){
    x<-li[[1]]
    li<-li[-1]
    if (is.element(list(x),li)){
      matchID <- match(list(x),li)+i
      cat(paste("Dopplung:",i,"matches:",matchID,"Jobs:",names(kodierungenV[[ks]])[i],"|",names(kodierungenV[[ks]])[matchID],"\n"))
    }
  }
  cat("\n")
}

for (ks in 1:6) {
  len <- length(kodierungenA[[ks]])
  li <- kodierungenA[[ks]]
  cat(paste0("Kategoriensystem: ",ks,"\n"))
  for (i in 1:len){
    #for (i in 1:50){
    x<-li[[1]]
    li<-li[-1]
    if (is.element(list(x),li)){
      matchID <- match(list(x),li)+i
      cat(paste("Dopplung:",i,"matches:",matchID,"Jobs:",names(kodierungenA[[ks]])[i],"|",names(kodierungenA[[ks]])[matchID],"\n"))
    }
  }
  cat("\n")
}

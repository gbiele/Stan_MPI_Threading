library(ggplot2)
library(data.table)
times = fread("results/all.txt", header = T)
times = times[sec > 1]
times[, model := factor(model, levels = c(0,1,2),
                       labels = c("basic","MPI","thereading"))
      ][,shards := factor(shards)
        ]

m = times[, list(sec = mean(sec)) , by = c("model","N","shards")]

ggplot(m, aes(x = N, y = sec, group = model:shards, color = model, lty = shards)) + geom_line() + geom_point()

m[, relative := sec/max(sec), by = "N"] 
ggplot(m, aes(x = N, y = relative, group = model:shards, color = model, lty = shards)) + geom_line() + geom_point()

m[, relative := sec/max(sec), by = "N"]
m[, relative2 := (1/as.numeric(as.character(shards)))/(sec/max(sec)), by = "N"] 
m[, analysis := paste0(shards,"-",model) ]


round(xtabs(relative~ analysis + N, m[model != "basic"]),digits = 2)

round(xtabs(relative2~ analysis + N, m[model != "basic"]),digits = 2)
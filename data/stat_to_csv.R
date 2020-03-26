library(haven)

files = c("election88/polls.dta", "election88/census88.dta", "election88/presvote.dta")

for (f in files) {
  yourData = read_dta(f)
  write.csv(yourData, file = gsub("dta", "csv", f))
}

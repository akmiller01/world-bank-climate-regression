# Setup ####
list.of.packages <- c("data.table", "rstudioapi")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")

mse = function(x, p){return(mean((x-p)^2))}

# Climate change total
dat = fread("output/wb_regression_inference.csv")
dat$`Climate change` = pmin(dat$`Climate change`, 1)
plot(dat$`Climate change`[order(dat$`Climate change`)])
hist(dat$`Climate change`)
plot(density(dat$`Climate change`))

plot(dat$pred[order(dat$pred)])
hist(dat$pred)
plot(density(dat$pred))

dat$pred_cap = pmax(dat$pred, 0)
dat$pred_cap = pmin(dat$pred_cap, 1)
plot(dat$pred_cap[order(dat$pred_cap)])
hist(dat$pred_cap)
plot(density(dat$pred_cap))

par(mfrow=(c(1, 2)))
plot(density(dat$`Climate change`))
plot(density(dat$pred_cap))
dev.off()

mse(dat$`Climate change`, mean(dat$`Climate change`))
mse(dat$`Climate change`, dat$pred_cap)
plot(`Climate change`~pred_cap, data=dat)

summary(lm(`Climate change`~pred_cap, data=dat))
mean(abs(dat$`Climate change` - dat$pred_cap))
mean(abs(dat$`Climate change` - dat$pred_cap) <= 0.1)
mean(abs(dat$`Climate change` - dat$pred_cap) <= 0.2)
mean(abs(dat$`Climate change` - dat$pred_cap) <= 0.3)

# Climate adaptation and mitigation model
dat$pred_a_cap = pmax(dat$pred_a, 0)
dat$pred_a_cap = pmin(dat$pred_a_cap, 1)

dat$pred_m_cap = pmax(dat$pred_m, 0)
dat$pred_m_cap = pmin(dat$pred_m_cap, 1)

mse(dat$`Climate adaptation`, mean(dat$`Climate adaptation`))
mse(dat$`Climate adaptation`, dat$pred_a)
mse(dat$`Climate adaptation`, dat$pred_a_cap)
plot(`Climate adaptation`~pred_a_cap, data=dat)

summary(lm(`Climate adaptation`~pred_a_cap, data=dat))
mean(abs(dat$`Climate adaptation` - dat$pred_a_cap))
mean(abs(dat$`Climate adaptation` - dat$pred_a_cap) <= 0.1)
mean(abs(dat$`Climate adaptation` - dat$pred_a_cap) <= 0.2)
mean(abs(dat$`Climate adaptation` - dat$pred_a_cap) <= 0.3)

mse(dat$`Climate mitigation`, mean(dat$`Climate mitigation`))
mse(dat$`Climate mitigation`, dat$pred_m)
mse(dat$`Climate mitigation`, dat$pred_m_cap)
plot(`Climate mitigation`~pred_m_cap, data=dat)

summary(lm(`Climate mitigation`~pred_m_cap, data=dat))
mean(abs(dat$`Climate mitigation` - dat$pred_m_cap))
mean(abs(dat$`Climate mitigation` - dat$pred_m_cap) <= 0.1)
mean(abs(dat$`Climate mitigation` - dat$pred_m_cap) <= 0.2)
mean(abs(dat$`Climate mitigation` - dat$pred_m_cap) <= 0.3)

# Check concurrance between labels
dat$pred_sum = dat$pred_a_cap + dat$pred_m_cap
dat$pred_sum = pmax(dat$pred_sum, 0)
dat$pred_sum = pmin(dat$pred_sum, 1)

plot(pred_cap~pred_sum, data=dat)
summary(lm(pred_cap~pred_sum, data=dat))
mse(dat$pred_cap, dat$pred_sum)
mse(dat$`Climate change`, dat$pred_cap)
mse(dat$`Climate change`, dat$pred_sum)
dat$between_labels_error = dat$pred_sum - dat$pred_cap
Hmisc::describe(dat$between_labels_error)
dat$between_reality_error = dat$`Climate change` - dat$pred_cap
plot(between_reality_error~between_labels_error, data=dat)
summary(lm(between_reality_error~between_labels_error, data=dat))


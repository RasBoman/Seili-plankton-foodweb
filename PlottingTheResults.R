library(tidyverse)
library(lubridate)
library(data.table)
library(rlist)
library(viridis)
library(RColorBrewer)


tiedostopolku <- "/Users/Ramu/GraduElokuu/MatlabVERSIOB"
tiedostonimet <- list.files(path = tiedostopolku, pattern = ".txt", full.names = T)

seili_biomass_for_plots <- read_csv(file = "/Users/heiditaskinen/documents/seili_biomass_log_for_plots.csv")
seili_nonlog_biomass_for_plots <- read_csv(file = "/Users/heiditaskinen/documents/seili_biomass_nonlog_for_plots.csv")

tiedostonimet_ei_polkua <- list.files(path = "/Users/Ramu/GraduElokuu/MatlabVERSIOB", 
                         pattern = ".txt",
                         full.names = F)

# putting all the results into a dataframe with the help of purrr:map
matlab_tulokset <- tiedostonimet %>% 
  map_df(~read.table(.)) %>%
  t()

quarters <- read_csv(file = "/Users/heiditaskinen/documents/quarters_and_years.csv", col_names = T)
quarters <- quarters %>%
  slice(3:103)

original_table_seili <- read_csv(file = "/Users/heiditaskinen/documents/seili_by_quarter_log_scaled_sliced.csv", col_names = T) %>%
  bind_cols(quarters) %>%
  select(quarter, year_e, -quarter1, everything())

# Taking the colnames from file names
colnames(matlab_tulokset) <-  c(tiedostonimet_ei_polkua)

#Rearranging and shortening the results for easier plotting
matlab_tulokset_vuosi_quarter <- matlab_tulokset %>%
  as_tibble %>%
  rowid_to_column("ID") %>%
  left_join(as_tibble(rowid_to_column(quarters, "ID")), by = "ID") %>% #lisää quarter & vuosi-sarakkeet
  select(quarter, 
         year_e, 
         ID,
         all_links_GenHVMu = Plankton_by_quarter_GenHVMu_No_time_VERSIOB.txt,
         all_links_GenHVSig = Plankton_by_quarter_GenHVSig_No_time_VERSIOB.txt,
         all_links_ZooHVMu = Plankton_by_quarter_ZooHVMu_No_time_VERSIOB.txt,
         all_links_ZooHVSig = Plankton_by_quarter_ZooHVSig_No_time_VERSIOB.txt,
         hv_linked_GenHVMu = Quarter_Plankton_GenHVMu_only_HV_linked_no_time_VERSIOB.txt,
         hv_linked_GenHVSig = Quarter_Plankton_GenHVSig_only_HV_linked_no_time_VERSIOB.txt,
         hv_linked_ZooHVMu = Quarter_Plankton_ZooHVMu_only_HV_linked_no_time_VERSIOB.txt,
         hv_linked_ZooHVSig = Quarter_Plankton_ZooHVSig_only_HV_linked_no_time_VERSIOB.txt)  %>%
  mutate(quarterwithyear = paste("Q", quarter, year_e, sep = " ")) %>%
  mutate(quarter = factor(quarter, levels = 1:4, labels = c("Winter","Spring","Summer", "Fall")))

# write_rds(matlab_tulokset_vuosi_quarter, "/Users/Ramu/GraduElokuu/OriginalData/matlab_tulokset.rds")

#Setting the theme for plotting
newtheme <- theme_bw() + theme(plot.title = element_text(size = 14, hjust = 0),
                               axis.text.x = element_text(size = 11, angle=45, hjust=0.9),
                               axis.text.y = element_text(size = 11),
                               axis.title.x = element_text(color = "grey20", size = 13, angle = 0, hjust = 0.5, vjust = 0.5, face = "plain"),
                               axis.title.y = element_text(color = "grey20", size = 13, angle = 90, hjust = 0.5, vjust = 0.5, face = "plain"))
theme_set(newtheme)

# A function for plotting the results from Bayesian model (mean and sd)
plot_seili_facet <- function(taulukko, HVMyy, HVSigma, PaaOtsikko, xOtsikko, yOtsikko, facet_plot = T){
  seili_plot <- taulukko %>%
    ggplot() +
    geom_ribbon(data = taulukko, aes(x = year_e,
                    ymin = HVMyy - (HVSigma / 2),
                    ymax = HVMyy + (HVSigma / 2)),
                fill = "grey70",
                alpha = 0.5) +
    geom_line(aes(x = year_e,
                  y = HVMyy)) +
    scale_x_continuous(breaks = seq(1990, 2016, by = 5),
                       minor_breaks = seq(1990, 2016, by = 1)) +
    labs(title = PaaOtsikko,
         x = xOtsikko,
         y = yOtsikko) +
    newtheme +
    facet_wrap(~quarter)
  
  print(seili_plot)
}

#Calling and saving each of the results
plotY <- plot_seili_facet(taulukko = matlab_tulokset_vuosi_quarter,
                 HVMyy = matlab_tulokset_vuosi_quarter$hv_linked_ZooHVMu,
                 HVSigma = matlab_tulokset_vuosi_quarter$hv_linked_ZooHVSig,
                 PaaOtsikko = "Model with only hidden variables linked across time slices",
                 xOtsikko = "Year",
                 yOtsikko = "Zooplanktonic hidden variable")

#ggsave("Zoo_HV_model_with_only_HV_linked.jpeg", 
#       device = "jpeg", 
#       units = "cm",
#       width = 24,
#       height = 18)

plot_genHV <- plot_seili_facet(taulukko = matlab_tulokset_vuosi_quarter,
                          HVMyy = matlab_tulokset_vuosi_quarter$hv_linked_GenHVMu,
                          HVSigma = matlab_tulokset_vuosi_quarter$hv_linked_GenHVSig,
                          PaaOtsikko = "Model with only hidden variables linked across time slices",
                          xOtsikko = "Year",
                          yOtsikko = "Generic hidden variable")

#ggsave("Gen_HV_model_with_only_HV_linked.jpeg", 
#       device = "jpeg", 
#       units = "cm",
#       width = 24,
#       height = 18)

plot_zooHV_autoreg <- plot_seili_facet(taulukko = matlab_tulokset_vuosi_quarter,
                               HVMyy = matlab_tulokset_vuosi_quarter$all_links_GenHVMu,
                               HVSigma = matlab_tulokset_vuosi_quarter$all_links_GenHVSig,
                               PaaOtsikko = "Model with phyto- and zooplankton as well as hidden variables linked across time slices",
                               xOtsikko = "Year",
                               yOtsikko = "Zooplanktonic hidden variable")

ggsave("Gen_HV_model_autoreg.jpeg", 
       device = "jpeg", 
       units = "cm",
       width = 24,
       height = 18)

plot_zooHV_autoreg <- plot_seili_facet(taulukko = matlab_tulokset_vuosi_quarter,
                                       HVMyy = matlab_tulokset_vuosi_quarter$all_links_ZooHVMu,
                                       HVSigma = matlab_tulokset_vuosi_quarter$all_links_ZooHVSig,
                                       PaaOtsikko = "Model with phyto- and zooplankton as well as hidden variables linked across time slices",
                                       xOtsikko = "Year",
                                       yOtsikko = "Zooplanktonic hidden variable")


ggsave("Zoo_HV_model_autoreg.jpeg", 
       device = "jpeg", 
       units = "cm",
       width = 24,
       height = 18)




################### Making the phytoplankton plots ####################

seili_phyto_for_plots <- seili_nonlog_biomass_for_plots %>%
  select(quarter,
         year_e,
         Diatomophyceae,
         Dinophyceae,
         Litostomatea,
         Cyanophyceae,
         Cryptophyceae,
         Chrysophyceae,
         Prymnesiophyceae) %>%
  mutate_all(~replace(., is.na(.), 0)) %>%
  gather(key = "Class", value = "Biomass", -quarter, -year_e) %>%
  mutate(quarterwithyear = paste("Q", quarter, year_e, sep = " ")) %>%
  mutate(quarter = factor(quarter, levels = 1:4, labels = c("Winter","Spring","Summer", "Fall")))


S_phyto_filtered <- seili_phyto_for_plots %>%
  group_by(quarter, year_e) %>%
  select(quarter, year_e, Class, Biomass) %>%
  filter(Biomass != 0)  %>%
  filter(Biomass != 0.759) 

#Use seili_phyto_for_plots for 0 values included 
phyto_plot <- S_phyto_filtered %>%
  ggplot(aes(x = year_e, y = Biomass / 1000000, fill = Class)) +
    geom_area() +
    facet_wrap(~quarter, scales = "free_y") +
    scale_fill_brewer(palette = "Dark2") +
    scale_x_continuous(breaks = seq(1990, 2016, by = 5),
                       minor_breaks = seq(1990, 2016, by = 1)) +
    labs(title = "Yearly succession of phytoplankton", 
         x = "Year",
         y = expression(paste("Mean biomass (", 10^3, mu, "g ", l^-1, ")", sep="")))# +
   # annotate("text", x = 2004, y = 0, label = "x", size = 2)

#Making data frames of missing data years
ann_winter <- data.frame(year_e = c(1990:1999, 2003:2006,2009:2011,2014:2016), 
                         Biomass = 0, 
                         lab = "NA",
                         quarter = "Winter", 
                         Class = NA)
ann_spring <- data.frame(year_e = c(1990:1991, 1994:1997, 2003, 2005, 2010), 
                         Biomass = 0, 
                         lab = "NA",
                         quarter = "Spring", 
                         Class = NA)

ann_summer <- data.frame(year_e = c(1990, 1995, 2005, 2007), 
                         Biomass = 0,
                         lab = "NA",
                         quarter = "Summer",
                         Class = NA)
ann_fall <- data.frame(year_e = c(1990, 1991, 1995:1997, 2005:2006, 2010, 2015:2016),
                       Biomass = 0,
                       lab = "NA",
                       quarter = "Fall",
                       Class = NA)
phyto_plot + 
  geom_text(data = ann_spring, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = ann_winter, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = ann_summer, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = ann_fall, label = "N/A", size = 1.6, angle = 75, hjust = 1)

ggsave("Yearly_succession_of_phytoplankton_NA_INCL.jpeg", 
       device = "jpeg", 
       units = "cm",
       width = 24,
       height = 18)

seili_zooplank_for_plots <- seili_nonlog_biomass_for_plots %>%
  select(quarter,
         year_e,
         Acartia = AcartiaTot,
         Daphnia = DaphniaTot,
         Eubosmina = Eubosmina_long, 
         Eurytemora = Eurytemora_aff, 
         Evadne = Evadne_normanni,
         Pleopsis = Pleopsis_polyp,
         Synchaeta = Synchaeta_sp) %>%
  mutate_all(~replace(., is.na(.), 0)) %>%
  gather(key = "Genus", value = "Biomass", -quarter, -year_e) %>%
  mutate(quarterwithyear = paste("Q", quarter, year_e, sep = " ")) %>%
  mutate(quarter = factor(quarter, levels = 1:4, labels = c("Winter","Spring","Summer", "Fall")))

S_zoo_filtered <- seili_zooplank_for_plots %>%
  group_by(quarter, year_e) %>%
  select(quarter, year_e, Genus, Biomass) %>%
#  filter(Biomass != 0) %>%
  filter(year_e > 1990) %>%
  filter(year_e < 2014) %>%
  filter(year_e != 1991|1992 & quarter != "Fall") %>%
  filter(!(year_e < 1997 & quarter == "Winter")) %>%
  filter(!(year_e == 2000 & quarter == "Winter")) %>%
  arrange(quarter, year_e)
  

print(S_zoo_filtered)

zplot <-S_zoo_filtered %>%
  ggplot(aes(x = year_e, y = Biomass / 1000, fill = Genus)) +
  geom_area() +
  facet_wrap(~quarter, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.text = element_text(face = "italic")) +
  scale_x_continuous(breaks = seq(1990, 2016, by = 5),
                     minor_breaks = seq(1990, 2016, by = 1)) +
  labs(title = "Yearly succession of zooplankton", 
       x = "Year",
       y = expression(paste("Mean biomass (", mu, "g ", l^-1, ")", sep="")))  # Tämä tarkastettu

z_ann_winter <- data.frame(year_e = c(1990:1996, 2000, 2014:2016), 
                         Biomass = 0, 
                         lab = "NA",
                         quarter = "Winter", 
                         Genus = NA)
z_ann_spring <- data.frame(year_e = c(1990:1991, 2014:2016), 
                         Biomass = 0, 
                         lab = "NA",
                         quarter = "Spring", 
                         Genus = NA)

z_ann_summer <- data.frame(year_e = c(1990, 2014:2016), 
                         Biomass = 0,
                         lab = "NA",
                         quarter = "Summer",
                         Genus = NA)
z_ann_fall <- data.frame(year_e = c(1990:1991, 2014:2016),
                       Biomass = 0,
                       lab = "NA",
                       quarter = "Fall",
                       Genus = NA)
zplot + 
  geom_text(data = z_ann_spring, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = z_ann_winter, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = z_ann_summer, label = "N/A", size = 1.6, angle = 75, hjust = 1) +
  geom_text(data = z_ann_fall, label = "N/A", size = 1.6, angle = 75, hjust = 1)


ggsave("Yearly_succession_of_zooplankton_NA_EXtr.jpeg", 
       device = "jpeg", 
       units = "cm",
       width = 24,
       height = 18)

# Plots of separate seasons

seili_zooplank_for_plots %>%
  filter(quarter == "Summer") %>%
  ggplot(aes(x = year_e, y = Biomass / 1000, fill = Genus)) +
  geom_area() +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.text = element_text(face = "italic")) +
  scale_x_continuous(breaks = seq(1990, 2016, by = 5),
                     minor_breaks = seq(1990, 2016, by = 1)) +
  labs(title = "Summer succession of zooplankton", 
       x = "Year",
       y = expression(paste("Mean biomass (", mu, "g ", l^-1, ")", sep="")))  # Tämä tarkastettu

ggsave("Summer_succession_of_zooplankton.jpeg", 
       device = "jpeg", 
       units = "cm",
       width = 24,
       height = 12)

### En ehkä lähtisi tuomaan ympäristömuuttujia tähän enää...

seili_env_variables_for_plots <- seili_nonlog_biomass_for_plots %>%
  select(quarter,
         Year = year_e,
         dis_org_nitr,
         dis_org_pho,
         Salinity = sal,
         Temperature = temp) %>%
  mutate_all(~replace(., is.na(.), 0))  %>%
  mutate(quarter = factor(quarter, levels = 1:4, labels = c("Winter","Spring","Summer", "Fall")))
  
seili_env_variables_for_plots %>%
  ggplot(aes(x = Year, y = Temperature)) +
    geom_line() +
    facet_wrap(~quarter)




########## TOISET TULOKSET I IGNORE? ##########
matlab_tulokset_vuosi_time_included <- matlab_tulokset %>%
  as_tibble %>%
  rowid_to_column("ID") %>%
  left_join(as_tibble(rowid_to_column(quarters, "ID")), by = "ID")  %>%
  select(quarter,
         year_e,
         ID,
         all_links_GenHVMu = Quarter_Plankton_GenHVMu_Dynamic_VERSIOB.txt,
         all_links_GenHVSig = Quarter_Plankton_GenHVSig_Dynamic_VERSIOB.txt,
         all_links_ZooHVMu = Quarter_Plankton_ZooHVMu_Dynamic_VERSIOB.txt,
         all_links_ZooHVSig = Quarter_Plankton_ZooHVSig_Dynamic_VERSIOB.txt,
         hv_linked_GenHVMu = Quarter_Plankton_GenHVMu_only_HV_linked_VERSIOB.txt,
         hv_linked_GenHVSig = Quarter_Plankton_GenHVSig_only_HV_linked_VERSIOB.txt,
         hv_linked_ZooHVMu = Quarter_Plankton_ZooHVMu_only_HV_linked_VERSIOB.txt,
         hv_linked_ZooHVSig = Quarter_Plankton_ZooHVSig_only_HV_linked_VERSIOB.txt) %>%
  mutate(QandY = paste(quarter, year_e))

matlab_tulokset_vuosi_quarter %>%
  ggplot(aes(x = ID, y = all_links_GenHVMu)) +
  geom_line() 

# Plotting temperature and salinity?

temp_plots <- read_rds("/Users/heiditaskinen/documents/MatLabResults/quarters_with_years.rds")
as.factor(temp_plots$quarter)

temp_plots %>%
  ggplot(aes(x = year_e, y = Saliniteetti)) +
    geom_line() +
    facet_wrap(~quarter)

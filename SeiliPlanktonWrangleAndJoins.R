# This script is used for wrangling the data foe the masters thesis 
# "Using dynamic Bayesian networks with hidden variables for change inference of the plankton community in the Archipelago Sea"
# The data used is unfortunately non-open data. For further info contact rasmus.a.boman@gmail.com
# As I look at the script nearly a year later, there seems to be a lot that could've been done more efficiently but...
# that'll do pig, that'll do.

#Libraries used to create the final table

library(scales)
library(fuzzyjoin)
library(tidyverse)
library(lubridate)
library(readxl)

##############################################################################################################################################
# Working with Zooplankton and biomass (wetweight) data
# setwd("/users/Ramu")

wet_weight <- read_csv("wetweight.csv") #Includes weights, original zooplanktons are individuals
seili_zoo <- read_xlsx("Seili_zoop_muok.xlsx", col_names =  T, skip = 1)

#Tidying up the Zooplankton data, prepare for wetweight join. Not the most elegant solution but should work.

tidy_zoo_pl <- seili_zoo %>%
  select(-Notes) %>% #Removing notes
  gather(key = "species", value = "indiv", -c("Date")) %>% #Gathering the data into tidy form
  mutate_at(vars(species), funs(str_replace_all(species, " ", "."))) %>% #Removing spaces and replacing with dots in species names
  mutate_at(vars(species), funs(str_replace_all(species, "-", "."))) %>% #Removing - and replacing with dots
  mutate_at(vars(species), funs(str_replace_all(species, "hamatus..", "hamatus."))) %>% #Removing double dots, centropages had these
  mutate_at(vars(indiv), funs(str_replace_all(indiv, ",", "."))) %>% #commas to dots
  mutate_at(vars(indiv), funs(as.numeric)) #variable as numeric

wet_weight <- wet_weight %>%
  rename(species = Spp) %>% #Changing one variable to be able to join
  select(-X) #Useless variable

#Joining tidyzoopl with wet weigths

tidy_zoo_joined <- tidy_zoo_pl %>%
  left_join(wet_weight, by = "species")

tidy_zoo_joined <- distinct(tidy_zoo_joined)   #Checking that all the names have matched and removing some duplicates (don't know why there are..)

(NAs_in_tidy <- (is.na(tidy_zoo_joined$wetweight_ug_ind))) #Checking which rows don't have a wetweight (i.e. the data not available for these species)
(unmatched <- tidy_zoo_joined[NAs_in_tidy,]) #Taking the row numbers out
(unmatched_species <- unique(unmatched$species)) #list of species with no wet_weight
# -> Unmatched wetweights are not Zooplankton (but larvaes and larger animals), so we'll leave them out

#Counting biomass of the zooplankton species

tidy_zoo_biomass <- tidy_zoo_joined %>%
  mutate(biomass_ug_m3 = wetweight_ug_ind * indiv) %>% #mikrograms (ug) in m3
  filter(!is.na(biomass_ug_m3)) %>% #Filtering out NA values from biomass 
  distinct #taking out distinct observations

#Spreading the data for further joins

tidy_zoo_biomass_bydate <- tidy_zoo_biomass %>%
  filter(!species %in% unmatched_species) %>% # filtering out unmatched species (the !zooplankton, (as checked on rows 42-44))
  distinct %>% #For some reason there has been duplicate values, removing them
  spread(key = species, value = biomass_ug_m3, fill  = 0) %>% #Spreading the data to variables (wide version), 
  select(-c("indiv", "wetweight_ug_ind")) %>% #Taking out individuals and wetweight as they're not needed anymore 
  group_by(Date) %>%
  summarise_if(is.numeric, max, na.rm = T) #If there are several observations, picking the max

seili_zoo_by_order <- tidy_zoo_biomass_bydate  %>%
  mutate_if(is.numeric, list(~na_if(., -Inf))) %>% #Replacing -Inf with NA
  mutate_if(is.numeric, funs(replace(., is.na(.), 0))) #Replacing NAs with 0, if there is no observations = no biomass (0 gr)
    
#Combining species to higher orders to simplify later work....

seili_zoo_order_biomass <- seili_zoo_by_order %>%
  transmute(Date = as_date(Date),
            year = year(Date),
            month = month(Date),
            day = day(Date),
            yday = yday(Date),
            AcartiaTot = rowSums(select(seili_zoo_by_order, starts_with("Acart")), na.rm = T),
            BalanusTot = rowSums(select(seili_zoo_by_order, contains("Bala")), na.rm = T),
            CalanoidaSp = rowSums(select(seili_zoo_by_order, contains("Cala")), na.rm = T),
            CentropagesTot = rowSums(select(seili_zoo_by_order, contains("Centrop")), na.rm = T),
            CyclopSp = rowSums(select(seili_zoo_by_order, starts_with("Cyclop")), na.rm = T),
            DaphniaTot = rowSums(select(seili_zoo_by_order, starts_with("Daphn")), na.rm = T),
            Eubosmina_long = rowSums(select(seili_zoo_by_order, starts_with("Eubos")), na.rm = T),
            Eurytemora_aff = rowSums(select(seili_zoo_by_order, starts_with("Euryte")), na.rm = T),
            Evadne_normanni = rowSums(select(seili_zoo_by_order, starts_with("Evadne")), na.rm = T),
            Keratella = rowSums(select(seili_zoo_by_order, starts_with("Kerate")), na.rm = T),
            Limnocalanus_mac = rowSums(select(seili_zoo_by_order, starts_with("Limno")), na.rm = T),
            Marenz_neg = rowSums(select(seili_zoo_by_order, starts_with("Marenz")), na.rm = T),
            Pleopsis_polyp = rowSums(select(seili_zoo_by_order, starts_with("Pleopsis")), na.rm = T),
            Podon_interm = rowSums(select(seili_zoo_by_order, starts_with("Podon")), na.rm = T),
            Synchaeta_sp = rowSums(select(seili_zoo_by_order, starts_with("Synch")), na.rm = T),
            Temora_sp = rowSums(select(seili_zoo_by_order, starts_with("Temora")), na.rm = T),
            Tintinnopsis_sp = rowSums(select(seili_zoo_by_order, starts_with("Tintin")), na.rm = T))

# Here further plotting and exploratory data analysis was made to pick the classes with largest biomasses

zoopl_by_biomass <- seili_zoo_order_biomass %>% # This is good tibble/df for joining environmental data and phytoplankton data
  select(Date,
         AcartiaTot,
         DaphniaTot,
         Eubosmina_long,
         Eurytemora_aff,
         Evadne_normanni,
         Pleopsis_polyp,
         Synchaeta_sp,
         year,
         month,
         day,
         yday
         )

##########################################################################################################################

#Bringing in the environmental variables

Seili_env <- read_xlsx("Seili_env_muok.xlsx")
glimpse(Seili_env)

#Picking out the most important variables for the tasks

seili_env_crucial <- Seili_env %>%
  rename(chl_ug_l = Chl_a_ug_l) %>%
  select(Date = Aika, temperature, NH4, NO3NO2N_ug_l, PO4, Naytesyvyys, Ptot_ug_l, Ntot_ug_l, chl_ug_l, Saliniteetti, Piidioksidi_mg_l) %>%
  mutate(sampl_depth = str_replace_all(Naytesyvyys, "0,0-10,0", "5"), #Turning longer hauls to average depths. 
         sampl_depth = str_replace_all(Naytesyvyys, "0,0-8,0", "5"), #These are mainly for chlorophyll so shouldn't matter in current study.
         sampl_depth = str_replace_all(Naytesyvyys, "0,0-6,0", "5"),
         sampl_depth = str_replace_all(Naytesyvyys, ",", "."), 
         sampl_depth = as.numeric(sampl_depth)) %>%
  select(-Naytesyvyys) #Removing old one

#Dropping observations > 10 m, and summarising the values as means in first 10 meters.

seili_env_crucial_means <- seili_env_crucial %>%
  mutate(year = year(Date), #adding dates, maybe helps in joining the tables...
         month = month(Date),
         day = day(Date),
         yday = yday(Date),
         Date = as_date(Date)) %>%
  filter(year > 1990, sampl_depth <= 10 | is.na(sampl_depth)) %>% #Removing observations from >10 meters as the plankton samples are from shallower water
  group_by(Date) %>%
  summarise_if(is.numeric, mean, na.rm =TRUE, na.action = na.pass) %>% #NAs not taken into account
  select(-sampl_depth)

##############################################################################################################################

#Bringing in phytoplankton observations

seili_phyto_muok <- read_xlsx("Seili_kpl_muok.xlsx")

#wrangling the variables a bit..

seili_phyto_cleaned <- seili_phyto_muok %>% 
  mutate(row_number = as.numeric(rownames(seili_phyto_muok)), # rowid_to_column should do the same...
         Date = dmy(Pvm),
         secchi_depth = Nakosyvyys / 1000000,
         biomassa_ug_l = as.numeric(gsub(",", ".", Biomassa_ug_l)),
         biomass_pros = as.numeric(gsub(",", ".", Biomassa_pros)) / 10000,
         carb_cont_ug_l = as.numeric(gsub(",", ".", Hiilisis_ug_l)),
         depth_range = as.factor(Syvyysvali)) %>%
  filter(!is.na(Laji)) %>% #removing rows with no data
  select(-Pvm, -Klo, -Kpl_l, -Biomassa_ug_l, -Biomassa_pros, -Hiilisis_ug_l, -Nayte_vol, -Syvyysvali, -Nakosyvyys) %>%
  select(Date, everything())

phytodates <- unique(seili_phyto_cleaned$Date) #212 (Dates of) observations.

#Summarising by class
seili_phyto_by_class <- seili_phyto_cleaned %>%
  group_by(Naytteenotto_Id, Luokka, depth_range) %>% #Id, Class of phytoplankton and depth_range
  summarise(Date = min(Date), #Always the same
            secchi_depth = min(secchi_depth),
            biomass_ug_l = sum(biomassa_ug_l), #Summing together the biomass
            biomass_perc = sum(biomass_pros), #Summing together the percentage
            carb_cont_ug_l = sum(carb_cont_ug_l)) %>% # ..And carbon content
  arrange(Date, Luokka)  

#Spreading the table

seili_phyto_for_joining <- seili_phyto_by_class %>%
  select(-carb_cont_ug_l, -biomass_perc) %>% #removing useless variables as they complicate the table..
  spread(key = "Luokka", value = biomass_ug_l)

colnames(seili_phyto_for_joining) #Checking the current classes of phytoplankton..

# Choosing biomass wise the most abundant classes

phyto_by_class_to_join <- seili_phyto_for_joining %>%
  select(1:3,
         Diatomophyceae,
         Dinophyceae,
         Litostomatea = 9,
         Cyanophyceae,
         Cryptophyceae,
         Chrysophyceae,
         Prymnesiophyceae) %>%
  mutate_at(vars(Diatomophyceae:Prymnesiophyceae), funs(ifelse(is.na(.), 0, .)))
  
# ###################################################################################################
# Joining zooplankton with environmental data..
# Joined with package:fuzzydates

#1. joining environmental data to zoo within +- 2 weeks, keeping just matches (205)!!

seili_env_zoo_joined <- zoopl_by_biomass %>% #left joining with +-2 weeks 
  difference_left_join(seili_env_crucial_means, by = "Date", max_dist = 14, distance_col = "join_dist_days") %>% #Creating a column to see how far the observations are from each other
  group_by(Date.x) %>% #Left join leaves multiple rows so let's pick the nearest by grouping Date.x
  slice(which.min(join_dist_days)) %>% #Taking just the nearest value
  select("Date.x":"Date.y", join_dist_days) #This keeps only variables to continue combining all the env data..

sum(duplicated(seili_env_zoo_joined$Date.x)) #Check whether some of the zoop dates are duplicated = nope.

# 2. taking ALL the dates out of environment data

seili_env_dates <- as_tibble(seili_env_crucial_means$Date)

# 3. joining ALL the env_dates with zooplankton data (STRICT JOIN) -> preparing for phytodata join

env_zoo_full_join <- seili_env_crucial_means %>% 
  left_join(seili_env_zoo_joined, by = c("Date" = "Date.y")) %>%
  mutate(date_e = Date,
         date_z = Date.x,
         dist_env_zoo = join_dist_days) %>%
  select(date_e, date_z, dist_env_zoo, everything()) %>%
  select(-Date, -Date.x, -year.x, -month.x, -day.x, -yday.x, -join_dist_days)

# 4. joining environmental data to PHYTO within +- 2 weeks, keeping just matches (205)!!

env_phyto_joined <- phyto_by_class_to_join %>% #left joining with +-2 weeks (commonly used)
  difference_left_join(seili_env_dates, by = c("Date" = "value"), max_dist = 14, distance_col = "dist_env_phyto") %>%
  group_by(Date) %>% #Left join leaves multiple rows so let's pick the nearest by grouping Date.x
  slice(which.min(dist_env_phyto))

### Checking before final joining...
 #2 env_dates have duplicated in zooplankton data and 4 in phytodata
sum(duplicated(env_zoo_full_join$date_e))
sum(duplicated(env_phyto_joined$value))

# 5. Combinining phytoenv with zooenv 

final_combo_join <- env_zoo_full_join %>%
  left_join(env_phyto_joined, by = c("date_e" = "value")) %>%
  rename(date_p = Date) %>%
  distinct(date_e, date_p, .keep_all = T) %>% #removing observations that got duplicated from phyto
  distinct(date_e, date_z, .keep_all = T) %>% #removing duplicate observations from zooplankton
  select(date_e, date_z, date_p, dist_env_zoo, dist_env_phyto, everything())

#Grouping by quarters to make more data for the model to operate

seili_by_quarter_intermediate_plot <- final_combo_join %>%
  mutate(month_e = month(date_e), # Separating months and years from the date
         year_e = year(date_e)) %>%
  mutate(quarter = case_when(    #Grouping different months to seasons
                    month_e %in% c(12, 1:2) ~ 1,
                    month_e %in% 3:5 ~ 2,
                    month_e %in% 6:8 ~ 3,
                    month_e %in% 9:11 ~ 4),
         hvgen = NA, # Adding hidden variables for matlab-models with no data
         hvzoo = NA) %>%
  select(quarter, 
         year_e, 
         everything(), 
         -c(date_e:dist_env_phyto, #Taking out unused variables.
            NH4, 
            Ptot_ug_l:chl_ug_l, 
            Piidioksidi_mg_l:yday,
            Naytteenotto_Id:depth_range,
            month_e)) %>%
  group_by(quarter, year_e) %>% #Grouping by quarter
  summarise_all(funs(mean(., na.rm=TRUE))) %>% # Taking means for each quarter
  arrange(year_e, quarter) 

# Writing rds to use in another script
# write_rds(seili_by_quarter_intermediate_plot, "/Users/heiditaskinen/documents/MatLabResults/quarters_with_years.rds")

# Separated to take the year out for plotting in a different table

seili_by_quarter_intermediate <- seili_by_quarter_intermediate_plot %>%
  select(quarter, # Putting these in the right order for matlab model
         dis_org_nitr = NO3NO2N_ug_l,
         dis_org_pho = PO4,
         sal = Saliniteetti,
         temp = temperature,
         hvgen,
         Diatomophyceae,
         Dinophyceae,
         Litostomatea,
         Cyanophyceae,
         Cryptophyceae,
         Chrysophyceae,
         Prymnesiophyceae,
         hvzoo,
         AcartiaTot,
         DaphniaTot,
         Eubosmina_long,
         Eurytemora_aff,
         Evadne_normanni,
         Pleopsis_polyp,
         Synchaeta_sp)

seili_by_quarter_intermediate$temp[21] <- 0.01 # This -0,3 Celsius water temperature messes up the scaling of variable. Changed to + celsisu

# Checked up til here 12.1.2020
# Log-scaling the variables for the model

seili_by_quarter_log_scaled <-  seili_by_quarter_intermediate %>% #The final amount of rowa
  mutate_at(.vars = 2:21, funs(log(.) %>% as.vector)) %>% #Turning the final table logarithmic
  mutate_all(~ replace(., . == -Inf, 0)) %>% #Replaceing -Inf as 0 
  mutate_at(.vars = 2:21, funs(scale(.) %>% as.vector)) #Standardising / scaling the variables

seili_by_quarter_log_scaled_sliced <- seili_by_quarter_log_scaled %>%
  ungroup() %>%
  slice(3:103) #Removing rows with mostly missing observations

seili_by_quarter_log_scaled_predict <- seili_by_quarter_log_scaled_sliced %>%
  ungroup() %>%
  slice(1:79)

full_df_for_preds <- seili_by_quarter_log_scaled_sliced %>%
  slice(1:91) 

bind_rows_and_predict_these <- seili_by_quarter_log_scaled_sliced %>% # Making the prediction data frame
  ungroup() %>%
  slice(80:91) %>%
  mutate(AcartiaTot = NA,
         DaphniaTot = NA,
         Eubosmina_long = NA,
         Eurytemora_aff = NA,
         Evadne_normanni = NA,
         Pleopsis_polyp = NA,
         Synchaeta_sp = NA) 

seili_predict_zoopl <- seili_by_quarter_log_scaled_predict %>%
  bind_rows(bind_rows_and_predict_these) # Adding them back into the dataframe with no values

seili_predict_future <- seili_predict_zoopl %>%
  slice(1:80) # Making another prediction set for predictions without additional data

write_csv(full_df_for_preds,
          path =  "/Users/heiditaskinen/documents/MatLabResults/seili_predict_true_results.csv")

write_csv(seili_predict_zoopl,
          path = "/Users/heiditaskinen/documents/seili_predict_zoopl.csv")

write_csv(seili_predict_future,
          path = "/Users/heiditaskinen/documents/seili_predict_future_80.csv")

write_csv(seili_by_quarter_log_scaled_sliced, 
          path = "/Users/heiditaskinen/documents/seili_by_quarter_log_scaled_sliced.csv")

write_csv(seili_by_quarter_log_scaled, 
          path = "/Users/heiditaskinen/documents/seili_by_quarter_log_scaled.csv")

######################################################################
#Below additional code for plotting purposes:
######################################################################

quarter_year_to_add_to_results <- final_combo_join %>%
  mutate(month_e = month(date_e),
         year_e = year(date_e)) %>%
  mutate(quarter = case_when(
    month_e %in% c(12, 1:2) ~ 1,
    month_e %in% 3:5 ~ 2,
    month_e %in% 6:8 ~ 3,
    month_e %in% 9:11 ~ 4),
    hvgen = NA,
    hvzoo = NA) %>%
  select(quarter, 
         year_e, 
         everything(), 
         -c(date_e:dist_env_phyto, 
            NH4, 
            Ptot_ug_l:chl_ug_l, 
            Piidioksidi_mg_l:yday,
            Naytteenotto_Id:depth_range,
            month_e)) %>%
  group_by(quarter, year_e) %>%
  summarise_all(funs(mean(., na.rm=TRUE))) %>%
  arrange(year_e, quarter) %>%
  select(quarter, year_e)

year_to_add_to_predictions <- quarter_year_to_add_to_results %>%
  ungroup() %>%
  slice(3:103) %>%
  slice(1:91) %>%
  mutate(q_year = paste("Q", quarter, " ", year_e, sep = ""))

write_csv(year_to_add_to_predictions,
          path = "/Users/heiditaskinen/documents/year_to_predictions.csv")

#Logarithmic version for plotting (not scaled)

seili_log_for_plotting <- seili_by_quarter_intermediate %>% 
  mutate_at(.vars = 2:21, funs(log(.) %>% as.vector)) %>%
  mutate_all(~ replace(., . == -Inf, 0)) %>%
  ungroup() %>%
  bind_cols(quarter_year_to_add_to_results) %>%
  mutate(yearANDquarter = paste("Q", quarter, year_e, sep = " ")) %>%
  select(-quarter1) %>%
  slice(3:103)

seili_reg_for_plotting <- seili_by_quarter_intermediate %>% 
  mutate_all(~ replace(., . == -Inf, 0)) %>%
  ungroup() %>%
  bind_cols(quarter_year_to_add_to_results) %>%
  mutate(yearANDquarter = paste("Q", quarter, year_e, sep = " ")) %>%
  select(-quarter1) %>%
  slice(3:103)

write_csv(seili_reg_for_plotting,
          path = "/Users/heiditaskinen/documents/seili_biomass_nonlog_for_plots.csv")

write_csv(seili_log_for_plotting,
          path = "/Users/heiditaskinen/documents/seili_biomass_log_for_plots.csv")

write_csv(quarter_year_to_add_to_results, 
          path = "/Users/heiditaskinen/documents/quarters_and_years.csv")

ais <- rep(1:4, length.out = 112)
quarter_year_to_add_to_results$quarter == ais

seili_by_quarter_scaled_WITH_YEAR <- final_combo_join %>%
  mutate(month_e = month(date_e),
         year_e = year(date_e)) %>%
  mutate(quarter = case_when(
    month_e %in% c(12, 1:2) ~ 1,
    month_e %in% 3:5 ~ 2,
    month_e %in% 6:8 ~ 3,
    month_e %in% 9:11 ~ 4),
    hvgen = NA,
    hvzoo = NA) %>%
  select(quarter, 
         year_e, 
         everything(), 
         -c(date_e:dist_env_phyto, 
            NH4, 
            Ptot_ug_l:chl_ug_l, 
            Piidioksidi_mg_l:yday,
            Naytteenotto_Id:depth_range,
            month_e)) %>%
  group_by(quarter, year_e) %>%
  summarise_all(funs(mean(., na.rm=TRUE))) %>%
  arrange(year_e, quarter) %>%
  select(year_e,
         quarter, # Putting these in the right order for matlab
         dis_org_nitr = NO3NO2N_ug_l,
         dis_org_pho = PO4,
         sal = Saliniteetti,
         temp = temperature,
         hvgen,
         Diatomophyceae,
         Dinophyceae,
         Litostomatea,
         Cyanophyceae,
         Cryptophyceae,
         Chrysophyceae,
         Prymnesiophyceae,
         hvzoo,
         AcartiaTot,
         DaphniaTot,
         Eubosmina_long,
         Eurytemora_aff,
         Evadne_normanni,
         Pleopsis_polyp,
         Synchaeta_sp) %>%
  mutate_at(.vars = 3:22, funs(log(.) %>% as.vector)) %>%
  mutate_all(~ replace(., . == -Inf, 0))

write_csv(seili_by_quarter_scaled_WITH_YEAR, 
          path = "/Users/heiditaskinen/documents/seili_by_quarter_scaled_WITH_YEAR.csv")


#Checking time differences between phyto, zoo and environmental values.

median(final_combo_join$dist_env_zoo, na.rm = T)
sum(final_combo_join$dist_env_phyto > 1, na.rm =T)
median(final_combo_join$dist_env_phyto, na.rm = T)
median(final_combo_join$dist_env_zoo - final_combo_join$dist_env_phyto, na.rm = T)

#90 "perfect" observations ----->

#Taking out double observations
# Not relevant?
phyto_zoo_matches <- final_combo_join[(!is.na(final_combo_join$date_z) | !is.na(final_combo_join$date_p)),]


write_rds(phyto_zoo_matches, path = "/Users/Ramu/GraduElokuu/EditedDataTables/combined_phyto_or_zoo_matches.rds")
write_excel_csv(phyto_zoo_matches, path = "/Users/Ramu/GraduElokuu/EditedDataTables/combined_phyto_or_zoo_matches.csv")


#############################################################################################################################
#DIFFERENT EXPLORATIONS FROM HERE ON, NOT RELEVANT!
#############################################################################################################################
##################################################################################################################################

### Zooplankton calculations taking just the 7 most frequent 

allZooplankties <- seili_zoo_order_biomass %>%
  select(-Date,
         -year,
         -month,
         -day,
         -yday) %>%
  gather(key = "species", value = "biomass") %>%
  select(-species) %>%
  colSums()

topSeven <- seili_zoo_order_biomass %>%
  select(AcartiaTot,
         DaphniaTot,
         Eubosmina_long,
         Eurytemora_aff,
         Evadne_normanni,
         Pleopsis_polyp,
         Synchaeta_sp) %>%
  gather(key = "species", value = "biomass") %>%
  select(-species) %>%
  colSums()

topSeven / allZooplankties
# Using the seven most frequent results to 97 % of the data. 

# Script below used for plotting later
seili_by_quarter_scaled_with_year <- final_combo_join %>%
  mutate(month_e = month(date_e),
         year_e = year(date_e)) %>%
  mutate(quarter = case_when(
    month_e %in% c(12, 1:2) ~ 1,
    month_e %in% 3:5 ~ 2,
    month_e %in% 6:8 ~ 3,
    month_e %in% 9:11 ~ 4),
    hvgen = NA,
    hvzoo = NA) %>%
  select(quarter, 
         year_e, 
         everything(), 
         -c(date_e:dist_env_phyto, 
            NH4, 
            Ptot_ug_l:chl_ug_l, 
            Piidioksidi_mg_l:yday,
            Naytteenotto_Id:depth_range,
            month_e)) %>%
  group_by(quarter, year_e) %>%
  summarise_all(funs(mean(., na.rm=TRUE))) %>%
  arrange(year_e, quarter) %>%
  select(quarter,
         year_e,# Putting these in the right order for matlab
         dis_org_nitr = NO3NO2N_ug_l,
         dis_org_pho = PO4,
         sal = Saliniteetti,
         temp = temperature,
         hvgen,
         Diatomophyceae,
         Dinophyceae,
         Litostomatea,
         Cyanophyceae,
         Cryptophyceae,
         Chrysophyceae,
         Prymnesiophyceae,
         hvzoo,
         AcartiaTot,
         DaphniaTot,
         Eubosmina_long,
         Eurytemora_aff,
         Evadne_normanni,
         Pleopsis_polyp,
         Synchaeta_sp) %>%
  mutate_at(.vars = 3:21, funs(scale(.) %>% as.vector))


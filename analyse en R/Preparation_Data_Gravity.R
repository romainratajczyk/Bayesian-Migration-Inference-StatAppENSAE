# ==============================================================================
# REPRODUCTION DU MODÈLE DE GRAVITÉ - VERSION ENSAE "FULL SPEC"
# ==============================================================================





# commentaire romain: API de l'ONU et banque mondiale pour les données IMR PSR et urban. IMR: à l'age 0?
# lire Santos Silva & Tenreyro, 2006 pour le débat économétrique flow+1



library(data.table)
library(readxl)
library(wpp2019)
library(countrycode)
library(WDI) # Pour l'urbanisation (SP.URB.TOTL.IN.ZS)
library(magrittr)

# 1. CHARGEMENT DES FICHIERS LOCAUX
top200      <- fread("ProjetStat/data/200isoRegionCodes.csv")
flows       <- fread("ProjetStat/data/abelCohen2019flowsv6_flowdt.csv")
dist_cepii  <- as.data.table(read_excel("ProjetStat/data/dist_cepii.xls"))
geo_cepii   <- as.data.table(read_excel("ProjetStat/data/geo_cepii.xls"))

flows[, `:=`(orig = toupper(orig), dest = toupper(dest), year0 = year0)]
top200[, iso := toupper(iso)]

# 2. MAPPING ISO3 UNIQUE
data(UNlocations)
iso_map <- as.data.table(UNlocations)[location_type == 4, .(country_code, name)]
iso_map[, iso3 := toupper(countrycode(country_code, "iso3n", "iso3c"))]
iso_map <- unique(iso_map[!is.na(iso3) & iso3 %in% top200$iso])

# 3. EXTRACTION DES DONNÉES PAYS (WPP & WDI)
data(popM); data(popF); data(mxM); data(mxF)
years_vec <- seq(1990, 2015, 5)
years_str <- as.character(years_vec)

# --- A. Population et PSR (wpp2019) ---
m_dt <- melt(as.data.table(popM), id.vars = c("country_code", "age"), measure.vars = years_str, variable.name = "year", value.name = "m")
f_dt <- melt(as.data.table(popF), id.vars = c("country_code", "age"), measure.vars = years_str, variable.name = "year", value.name = "f")
m_dt[, year := as.numeric(as.character(year))]; f_dt[, year := as.numeric(as.character(year))]

country_stats <- merge(m_dt, f_dt, by = c("country_code", "age", "year"))[, .(tot = m + f), by = .(country_code, age, year)]
country_stats <- country_stats[, .(
  P_t = sum(tot), 
  psr = sum(tot[age %in% c("15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64")]) / 
    sum(tot[age %in% c("65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+")])
), by = .(country_code, year)]

# --- B. Mortalité Infantile (IMR) ---
mx_cols <- grep("1990|1995|2000|2005|2010|2015", names(mxM), value = TRUE)
imr_dt <- merge(melt(as.data.table(mxM)[age == 0], id.vars = "country_code", measure.vars = mx_cols, value.name = "imr_m"),
                melt(as.data.table(mxF)[age == 0], id.vars = "country_code", measure.vars = mx_cols, value.name = "imr_f"),
                by = c("country_code", "variable"))
imr_dt[, `:=`(year = as.numeric(substr(variable, 1, 4)), IMR_t = (imr_m + imr_f) / 2)]
country_stats <- merge(country_stats, imr_dt[, .(country_code, year, IMR_t)], by = c("country_code", "year"))

# --- C. Urbanisation (WDI) ---
urban_raw <- as.data.table(WDI(indicator = "SP.URB.TOTL.IN.ZS", start = 1990, end = 2015, extra = FALSE))
urban_raw[, iso3 := toupper(countrycode(iso2c, "iso2c", "iso3c"))]
urban_raw <- urban_raw[year %in% years_vec, .(iso3, year, urban_t = SP.URB.TOTL.IN.ZS)]
# On fusionne l'urbanisation via l'iso3 dans le mapping pays
country_stats <- merge(country_stats, iso_map[, .(country_code, iso3)], by = "country_code")
country_stats <- merge(country_stats, urban_raw, by = c("iso3", "year"), all.x = TRUE)

# --- D. LA et LL (geo_cepii) ---
geo_clean <- geo_cepii[, .(LA = mean(area, na.rm=TRUE), LL = max(landlocked, na.rm=TRUE)), by = .(iso3 = toupper(iso3))]
country_stats <- merge(country_stats, geo_clean, by = "iso3")

# 4. ASSEMBLAGE FINAL (DOUBLE JOIN SÉCURISÉ)
master_dt <- flows[orig %in% top200$iso & dest %in% top200$iso]

# Ajout des codes numériques
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)], by.x = "orig", by.y = "iso3") %>% setnames("country_code", "cod_o")
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)], by.x = "dest", by.y = "iso3") %>% setnames("country_code", "cod_d")

# Doubles jointures Origine/Destination avec all.x = TRUE (POUR GARDER LES ZÉROS)
master_dt <- merge(master_dt, country_stats, by.x = c("cod_o", "year0"), by.y = c("country_code", "year"), all.x = TRUE)
setnames(master_dt, c("P_t", "psr", "IMR_t", "LA", "LL", "urban_t"), c("P_it", "PSR_i", "IMR_it", "LA_i", "LL_i", "urban_it"))

master_dt <- merge(master_dt, country_stats, by.x = c("cod_d", "year0"), by.y = c("country_code", "year"), all.x = TRUE)
setnames(master_dt, c("P_t", "psr", "IMR_t", "LA", "LL", "urban_t"), c("P_jt", "PSR_j", "IMR_jt", "LA_j", "LL_j", "urban_jt"))

# Jointure Bilatérale CEPII (Distances & Dummies)
dist_clean <- dist_cepii[, .(D_ij = mean(distcap, na.rm=TRUE), LB_ij = max(contig, na.rm=TRUE), 
                             OL_ij = max(comlang_off, na.rm=TRUE), COL_ij = max(colony, na.rm=TRUE)), 
                         by = .(iso_o = toupper(iso_o), iso_d = toupper(iso_d))]
master_dt <- merge(master_dt, dist_clean, by.x = c("orig", "dest"), by.y = c("iso_o", "iso_d"), all.x = TRUE)
# 5. TRANSFORMATIONS LOGS & CRÉATION DES 4 CIBLES
master_dt[, `:=`(
  t_2000 = year0 - 2000, 
  t_2000_sq = (year0 - 2000)^2,
  
  # --- LES 4 ALTERNATIVES DEMANDÉES ---
  # 1. Vérité Terrain (Flux brut avec Zéros)
  flow_raw = flow, 
  
  # 2. Donnée "Sale" (Log classique + 1)
  log_flow_plus_1 = log(flow + 1),
  
  # 3. Alternative Robuste (Inverse Hyperbolic Sine - gère le 0 proprement)
  ihs_flow = asinh(flow),
  
  # 4. Binaire (Migration ou pas ? 1 si >0, sinon 0)
  is_migration = as.integer(flow > 0)
)]

# CRUCIAL : On ne filtre PLUS "[flow > 0]" pour garder les zéros dans le fichier
gravity_ready <- master_dt

fwrite(gravity_ready, 
       file = "ProjetStat/data/gravity_data_WR_replique.csv", 
       sep = ",", 
       dec = ".", 
       row.names = FALSE, 
       col.names = TRUE)
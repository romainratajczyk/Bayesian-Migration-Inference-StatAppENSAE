# ==============================================================================
# REPRODUCTION DU MODÈLE DE GRAVITÉ - VERSION ENSAE "PIB + LAG"
# ==============================================================================
install.packages("wpp2019")
install.packages("countrycode")
install.packages("WDI")
install.packages("magrittr")
#---------------------------------------------------
library(data.table)
library(readxl)
library(wpp2019)
library(countrycode)
library(WDI) 
library(magrittr)

# 0. OPTIONS DE SÉCURITÉ ONYXIA
options(warn = 0)

# 1. CHARGEMENT DES FICHIERS LOCAUX
top200      <- fread("ProjetStat/data/200isoRegionCodes.csv")
flows       <- fread("ProjetStat/data/abelCohen2019flowsv6_flowdt.csv")
dist_cepii  <- as.data.table(read_excel("ProjetStat/data/dist_cepii.xls"))
geo_cepii   <- as.data.table(read_excel("ProjetStat/data/geo_cepii.xls"))


# --- INSERTION ICI : HARMONISATION ROUMANIE ---
for (dt in list(flows, top200, dist_cepii, geo_cepii)) {
  cols <- names(dt)[sapply(dt, is.character)]
  for (j in cols) {
    dt[get(j) %in% c("ROM", "RO"), (j) := "ROU"]
  }
}

flows[, `:=`(orig = toupper(orig), dest = toupper(dest), year0 = year0)]
top200[, iso := toupper(iso)]

# 2. MAPPING ISO3 UNIQUE
data(UNlocations)
iso_map <- as.data.table(UNlocations)[location_type == 4, .(country_code, name)]
iso_map[, iso3 := toupper(suppressWarnings(countrycode(country_code, "iso3n", "iso3c")))]
iso_map <- unique(iso_map[!is.na(iso3) & iso3 %in% top200$iso])

# 3. EXTRACTION DES DONNÉES PAYS
data(popM); data(popF); data(mxM); data(mxF)
years_vec <- seq(1990, 2015, 5)
years_str <- as.character(years_vec)

# --- A. Population et PSR ---
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

# --- C. Urbanisation & PIB (WDI) ---
# Indicateurs : Urbanisation et PIB courant (NY.GDP.MKTP.CD)
wdi_raw <- as.data.table(WDI(indicator = c("urban" = "SP.URB.TOTL.IN.ZS", "gdp" = "NY.GDP.MKTP.CD"), 
                             start = 1989, end = 2015, extra = FALSE))

setnames(wdi_raw, 
         old = c("iso3c", "SP.URB.TOTL.IN.ZS", "NY.GDP.MKTP.CD"), 
         new = c("iso3", "urban", "gdp"), 
         skip_absent = TRUE)

# CORRECTION ICI : WDI renvoie 'iso3c' et non 'iso3'
wdi_raw <- wdi_raw[!is.na(iso3) & iso3 %in% top200$iso]

# Création du Lag PIB : On décale l'année de +1 pour que la valeur de 1989 matche avec 1990
gdp_lag <- wdi_raw[, .(iso3, year = year + 1, PIB_lag = gdp)]
wdi_final <- merge(wdi_raw[year %in% years_vec, .(iso3, year, urban_t = urban, PIB = gdp)], 
                   gdp_lag, by = c("iso3", "year"), all.x = TRUE)

# Fusion dans country_stats
country_stats <- merge(country_stats, iso_map[, .(country_code, iso3)], by = "country_code")
country_stats <- merge(country_stats, wdi_final, by = c("iso3", "year"), all.x = TRUE)

# --- D. LA et LL (geo_cepii) ---
geo_clean <- geo_cepii[, .(LA = mean(area, na.rm=TRUE), LL = max(landlocked, na.rm=TRUE)), by = .(iso3 = toupper(iso3))]
country_stats <- merge(country_stats, geo_clean, by = "iso3", all.x = TRUE)

# 4. ASSEMBLAGE FINAL (DOUBLE JOIN)
master_dt <- flows[orig %in% top200$iso & dest %in% top200$iso]
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)], by.x = "orig", by.y = "iso3", all.x = TRUE) %>% setnames("country_code", "cod_o")
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)], by.x = "dest", by.y = "iso3", all.x = TRUE) %>% setnames("country_code", "cod_d")

# CORRECTION DANS LES SETNAMES (Attention aux doublons de noms)
master_dt <- merge(master_dt, country_stats, by.x = c("cod_o", "year0"), by.y = c("country_code", "year"), all.x = TRUE)
setnames(master_dt, c("P_t", "psr", "IMR_t", "LA", "LL", "urban_t", "PIB", "PIB_lag"), 
         c("P_it", "PSR_i", "IMR_it", "LA_i", "LL_i", "urban_it", "gdp_o", "gdp_o_lag")) # Corrigé : gdp_o et gdp_o_lag

master_dt <- merge(master_dt, country_stats, by.x = c("cod_d", "year0"), by.y = c("country_code", "year"))
setnames(master_dt, c("P_t", "psr", "IMR_t", "LA", "LL", "urban_t", "PIB", "PIB_lag"), 
         c("P_jt", "PSR_j", "IMR_jt", "LA_j", "LL_j", "urban_jt", "gdp_d", "gdp_d_lag"))
# Jointure Bilatérale CEPII
dist_clean <- dist_cepii[, .(D_ij = mean(distcap, na.rm=TRUE), LB_ij = max(contig, na.rm=TRUE), 
                             OL_ij = max(comlang_off, na.rm=TRUE), COL_ij = max(colony, na.rm=TRUE)), 
                         by = .(iso_o = toupper(iso_o), iso_d = toupper(iso_d))]
master_dt <- merge(master_dt, dist_clean, by.x = c("orig", "dest"), by.y = c("iso_o", "iso_d"), all.x = TRUE)

# 5. VARIABLES TEMPORELLES ET FILTRAGE
master_dt[, `:=`(
  t_2000 = year0 - 2000, 
  t_2000_sq = (year0 - 2000)^2
)]

gravity_ready <- master_dt
gravity_ready <- master_dt[!is.na(cod_o) & !is.na(cod_d)]

# 1. Correctif de sécurité pour la Roumanie (ROM -> ROU) sur les identifiants
gravity_ready[orig == "ROM", orig := "ROU"]
gravity_ready[dest == "ROM", dest := "ROU"]

# 2. Création des variables de flux et identifiants demandés
gravity_ready[, `:=`(
  year = year0,
  iso3_o = orig,
  iso3_d = dest,
  flow_raw = flow,
  log_flow_plus_1 = log1p(flow),
  ihs_flow = asinh(flow),
  is_migration = as.integer(flow > 0)
)]

# 3. Création des variables de PIB par tête (GDPCAP)
gravity_ready[, `:=`(
  gdpcap_o = gdp_o / P_it,
  gdpcap_d = gdp_d / P_jt,
  gdpcap_o_lag = gdp_o_lag / P_it, # Approximation via la population au temps t
  gdpcap_d_lag = gdp_d_lag / P_jt
)]

# 4. Calcul automatique des 8 logs pour les variables de richesse
gdp_vars <- c("gdp_o", "gdpcap_o", "gdp_d", "gdpcap_d", 
              "gdp_o_lag", "gdpcap_o_lag", "gdp_d_lag", "gdpcap_d_lag")

gravity_ready[, (paste0("log_", gdp_vars)) := lapply(.SD, function(x) log(x)), .SDcols = gdp_vars]

# 5. Sélection et ordonnancement final des 44 colonnes
final_cols <- c(
  'orig', 'dest', 'iso3_d', 'year', 'iso3_o', 'flow', 'P_it', 'PSR_i', 
  'IMR_it', 'urban_it', 'LA_i', 'LL_i', 'P_jt', 'PSR_j', 'IMR_jt', 
  'urban_jt', 'LA_j', 'LL_j', 'D_ij', 'LB_ij', 'OL_ij', 'COL_ij', 
  't_2000', 't_2000_sq', 'flow_raw', 'log_flow_plus_1', 'ihs_flow', 
  'is_migration', 'gdp_o', 'gdpcap_o', 'gdp_d', 'gdpcap_d', 'gdp_o_lag', 
  'gdpcap_o_lag', 'gdp_d_lag', 'gdpcap_d_lag', 'log_gdp_o', 'log_gdp_d', 
  'log_gdpcap_o', 'log_gdpcap_d', 'log_gdp_o_lag', 'log_gdp_d_lag', 
  'log_gdpcap_o_lag', 'log_gdpcap_d_lag'
)

df_final <- gravity_ready[, ..final_cols]

# 6. EXPORTATION
fwrite(df_final, 
       file = "ProjetStat/data/FINAL_GRAVITY_TRAINING_MATRIX.csv", 
       sep = ",", dec = ".", row.names = FALSE, col.names = TRUE)

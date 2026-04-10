import pandas as pd
import numpy as np
from pandas_datareader import wb
import requests
import io

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FLOW_FILE = 'bayesFlow/local/data/azoseRaftery2019flows.csv'
URL_CEPII = "https://github.com/aziizsut/analysis_global_energy/raw/master/dist_cepii.xls"
TARGET_YEARS = [1990, 1995, 2000, 2005, 2010]

# ==============================================================================
# ÉTAPE 1 : ANALYSE ET FILTRAGE DES FLUX
# ==============================================================================
print("--- ÉTAPE 1 : Nettoyage de l'Univers des Flux ---")

# 1. Chargement
df_flows = pd.read_csv(FLOW_FILE)
print(f"Lignes totales initiales : {len(df_flows)}")

# 2. Suppression des zéros (Strictement positif)
df_flows_active = df_flows[df_flows['migrantCount'] > 0].copy()
print(f"Lignes actives (migrantCount > 0) : {len(df_flows_active)}")

# 3. Extraction de l'Univers Raftery (Origine + Destination)
raftery_universe = set(df_flows_active['origIso'].unique()) | set(df_flows_active['destIso'].unique())
print(f"Nombre de pays uniques dans l'univers Raftery actif : {len(raftery_universe)}")

# ==============================================================================
# ÉTAPE 2 : AUDIT DES CODES ISO (RAFTERY vs BANQUE MONDIALE)
# ==============================================================================
print("\n--- ÉTAPE 2 : Audit ISO & Définition des Corrections ---")

# 1. Récupérer les codes officiels WB
try:
    wb_countries = wb.get_countries()
    wb_iso_set = set(wb_countries['iso3c'])
except:
    print("Erreur connexion WB. On continue avec les corrections manuelles connues.")
    wb_iso_set = set()

# 2. Comparaison : Qui est dans Raftery mais inconnu de la WB ?
missing_in_wb = raftery_universe - wb_iso_set
print(f"Codes Raftery inconnus de la WB (avant correction) : {sorted(list(missing_in_wb))}")

# 3. Dictionnaire de Mapping (Raftery -> WB) pour le téléchargement
# Ce dictionnaire dit : "Pour avoir les données de X (Raftery), demande à WB le code Y"
iso_mapping_download = {
    'ROM': 'ROU',  # Roumanie
    'ZAR': 'COD',  # R.D. Congo
    'TMP': 'TLS',  # Timor-Leste
    'FYR': 'MKD',  # Macédoine du Nord
    'WBG': 'PSE',  # Palestine
    'KOS': 'XKX',  # Kosovo
    # Les codes identiques restent identiques implicitement, sauf si listés ici
}

# 4. Stratégie des Proxies (Territoires sans données WB -> Utiliser le parent)
proxies = {
    'GLP': 'FRA', 'MTQ': 'FRA', 'GUF': 'FRA', 'REU': 'FRA', 'MYT': 'FRA', # France
    'CLI': 'GBR', # Channel Islands -> UK
    'SSD': 'SDN', # South Sudan -> Sudan (pour les années anciennes)
    'MNE': 'SRB', # Montenegro -> Serbia
    'GUM': 'USA', 'VIR': 'USA' # US Territories -> USA
}

# ==============================================================================
# ÉTAPE 3 : TÉLÉCHARGEMENT MONADIQUE (DÉMOGRAPHIE)
# ==============================================================================
print("\n--- ÉTAPE 3 : Construction de la Base Monadique ---")

indicators = {
    'SP.POP.TOTL': 'pop',
    'SP.POP.1564.TO.ZS': 'pop_15_64_pct',
    'SP.POP.65UP.TO.ZS': 'pop_65_plus_pct',
    'SP.DYN.IMRT.IN': 'IMR',
    'SP.URB.TOTL.IN.ZS': 'urban',
    'AG.LND.TOTL.K2': 'LA'
}

# Liste de tous les codes WB nécessaires (les normaux + les corrections + les parents des proxies)
wb_query_list = list(raftery_universe.intersection(wb_iso_set)) # Ceux qui matchent déjà
wb_query_list += list(iso_mapping_download.values()) # Les corrections (ROU, COD...)
wb_query_list += list(proxies.values()) # Les parents (FRA, GBR...)
wb_query_list = list(set(wb_query_list)) # Déduplication

print(f"Interrogation de l'API WB pour {len(wb_query_list)} codes...")

dfs = []
for code, name in indicators.items():
    try:
        # Téléchargement par lot pour éviter timeout "all", mais plus rapide que 1 par 1
        d = wb.download(indicator=code, country=wb_query_list, start=1990, end=2015)
        d = d.reset_index()
        d = d.rename(columns={code: name})
        d['year'] = d['year'].astype(int)
        d = d.set_index(['country', 'year'])
        dfs.append(d)
    except Exception as e:
        print(f"   Erreur sur {name}: {e}")

if dfs:
    df_wb = pd.concat(dfs, axis=1).reset_index()
    df_wb = df_wb.loc[:, ~df_wb.columns.duplicated()]

    # Mapping ISO WB
    iso_map_wb = wb_countries.set_index('name')['iso3c'].to_dict()
    df_wb['iso3_wb'] = df_wb['country'].map(iso_map_wb)

    # --- PHASE CRITIQUE : RETOUR AUX CODES RAFTERY ---
    # On doit transformer les données WB (ROU) en format Raftery (ROM)

    final_rows = []

    # Inversion du mapping download pour retrouver la clé Raftery
    # wb_code (ROU) -> raftery_code (ROM)
    wb_to_raftery = {v: k for k, v in iso_mapping_download.items()}

    for iso_target in raftery_universe:

        # Cas 1 : Le pays est un Proxy (ex: GLP)
        if iso_target in proxies:
            parent = proxies[iso_target]
            # On prend les données du parent
            subset = df_wb[df_wb['iso3_wb'] == parent].copy()
            subset['iso3'] = iso_target # On le renomme GLP
            final_rows.append(subset)

        # Cas 2 : Le pays a un mapping spécial (ex: ROM)
        elif iso_target in iso_mapping_download:
            wb_code = iso_mapping_download[iso_target]
            subset = df_wb[df_wb['iso3_wb'] == wb_code].copy()
            subset['iso3'] = iso_target # On renomme ROU -> ROM
            final_rows.append(subset)

        # Cas 3 : Le pays est standard (ex: FRA)
        else:
            subset = df_wb[df_wb['iso3_wb'] == iso_target].copy()
            subset['iso3'] = iso_target
            final_rows.append(subset)

    df_monadic = pd.concat(final_rows, ignore_index=True)

    # Filtrage Temporel
    df_monadic = df_monadic[df_monadic['year'].isin(TARGET_YEARS)].copy()

    # Déduplication et Calculs
    cols_num = list(indicators.values())
    df_monadic = df_monadic.groupby(['iso3', 'year'])[cols_num].mean().reset_index()

    # Calcul PSR
    df_monadic['PSR'] = np.where(df_monadic['pop_65_plus_pct']>0, df_monadic['pop_15_64_pct']/df_monadic['pop_65_plus_pct'], 0)

    # Export
    df_monadic.to_csv('/content/ProjetStat/data_final/gravity_monadic_covariates_CLEAN.csv', index=False)
    print(f"✅ Base Monadique générée : {len(df_monadic)} lignes (Cible: {len(raftery_universe)*5})")

    # Check manquants
    missing_monadic = raftery_universe - set(df_monadic['iso3'].unique())
    if missing_monadic:
        print(f"⚠️ Pays monadiques encore manquants : {missing_monadic}")


# ==============================================================================
# ÉTAPE 4 : TÉLÉCHARGEMENT DYADIQUE (CEPII)
# ==============================================================================
print("\n--- ÉTAPE 4 : Construction de la Base Dyadique ---")

try:
    df_cepii = pd.read_excel(URL_CEPII)

    cols_map = {'iso_o': 'iso_o', 'iso_d': 'iso_d', 'distcap': 'distcap', 'contig': 'contig', 'comlang_off': 'comlang_off', 'colony': 'col_dep_ever'}
    if 'colony' not in df_cepii.columns and 'col45' in df_cepii.columns: cols_map['col45'] = 'col_dep_ever'

    df_dy = df_cepii[list(cols_map.keys())].rename(columns=cols_map)

    # MAPPING CEPII -> RAFTERY
    # CEPII utilise souvent ROM, ZAR (comme Raftery), mais parfois des codes différents
    # On force l'alignement
    cepii_patches = {
        'ROU': 'ROM', 'COD': 'ZAR', 'MKD': 'FYR', 'TLS': 'TMP', 'PSE': 'WBG'
    }
    df_dy['iso_o'] = df_dy['iso_o'].replace(cepii_patches)
    df_dy['iso_d'] = df_dy['iso_d'].replace(cepii_patches)

    # GÉNÉRATION DE PROXIES DYADIQUES (Si un pays Raftery n'est pas dans CEPII)
    # Ex: SSD n'est pas dans CEPII -> On clone SDN
    existing_dy_isos = set(df_dy['iso_o'].unique()) | set(df_dy['iso_d'].unique())
    missing_dy_isos = raftery_universe - existing_dy_isos

    new_dy_rows = []
    dy_proxies = {'SSD': 'SDN', 'MNE': 'SRB', 'GUM': 'USA', 'VIR': 'USA', 'SXM': 'NLD', 'CUW': 'NLD'}

    for target in missing_dy_isos:
        if target in dy_proxies:
            src = dy_proxies[target]
            # Cloner Origine
            clone_o = df_dy[df_dy['iso_o'] == src].copy(); clone_o['iso_o'] = target
            # Cloner Destination
            clone_d = df_dy[df_dy['iso_d'] == src].copy(); clone_d['iso_d'] = target

            new_dy_rows.extend([clone_o, clone_d])

    if new_dy_rows:
        df_dy = pd.concat([df_dy] + new_dy_rows, ignore_index=True)

    # FILTRAGE FINAL & EXPANSION
    df_dy = df_dy[df_dy['iso_o'].isin(raftery_universe) & df_dy['iso_d'].isin(raftery_universe)]
    df_dy = df_dy.drop_duplicates(subset=['iso_o', 'iso_d'])

    dfs_years = []
    for y in TARGET_YEARS:
        t = df_dy.copy()
        t['year'] = y
        dfs_years.append(t)

    df_dy_final = pd.concat(dfs_years)
    df_dy_final.to_csv('/content/ProjetStat/data_final/gravity_dyadic_covariates_CLEAN.csv', index=False)

    print(f"✅ Base Dyadique générée : {len(df_dy_final)} lignes")

except Exception as e:
    print(f"Erreur CEPII : {e}")

print("\n--- TERMINÉ : Prêt pour la fusion finale ---")
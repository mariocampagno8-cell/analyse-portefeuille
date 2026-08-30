"""
Univers de valeurs et conventions de tickers Yahoo Finance.

Yahoo identifie une valeur par un code suivi d'un suffixe de place. La meme
societe cotee sur deux places a deux tickers differents, avec des devises et
des historiques distincts. Toujours verifier sur finance.yahoo.com avant
d'ajouter un ticker inconnu.
"""

from __future__ import annotations

# ==========================================================================
# Suffixes de place
# ==========================================================================

SUFFIXES = {
    "": "États-Unis (NYSE, Nasdaq)",
    ".PA": "Euronext Paris",
    ".AS": "Euronext Amsterdam",
    ".BR": "Euronext Bruxelles",
    ".LS": "Euronext Lisbonne",
    ".MI": "Borsa Italiana (Milan)",
    ".MC": "Bolsa de Madrid",
    ".DE": "Xetra (Francfort)",
    ".F": "Francfort",
    ".SW": "SIX Swiss Exchange",
    ".VI": "Vienne",
    ".L": "London Stock Exchange",
    ".ST": "Stockholm",
    ".OL": "Oslo",
    ".CO": "Copenhague",
    ".HE": "Helsinki",
    ".TO": "Toronto",
    ".V": "TSX Venture",
    ".T": "Tokyo",
    ".HK": "Hong Kong",
    ".SS": "Shanghai",
    ".SZ": "Shenzhen",
    ".KS": "Corée (KOSPI)",
    ".TW": "Taïwan",
    ".NS": "Inde (NSE)",
    ".BO": "Inde (BSE)",
    ".AX": "Australie (ASX)",
    ".NZ": "Nouvelle-Zélande",
    ".SA": "Brésil (B3)",
    ".MX": "Mexique",
    ".JO": "Johannesburg",
    ".TA": "Tel-Aviv",
    ".IS": "Istanbul",
    ".SI": "Singapour",
}

EXEMPLES_FORMATS = {
    "Action française": "AIR.PA (Airbus), MC.PA (LVMH), TTE.PA (TotalEnergies)",
    "Action américaine": "AAPL, MSFT, BRK-B (le tiret remplace le point)",
    "Action allemande": "SAP.DE, SIE.DE, ALV.DE",
    "Indice": "^FCHI (CAC 40), ^GSPC (S&P 500), ^STOXX50E, ^N225",
    "ETF européen": "IWDA.AS, CW8.PA, EUNL.DE",
    "Devise": "EURUSD=X, USDJPY=X, EURCHF=X",
    "Matière première": "GC=F (or), CL=F (pétrole WTI), SI=F (argent)",
    "Crypto": "BTC-EUR, ETH-USD, SOL-EUR",
    "Obligation": "^TNX (10 ans US), ^FVX (5 ans US)",
}


# ==========================================================================
# Univers predefinis
# ==========================================================================

CAC40 = [
    "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA", "CAP.PA",
    "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EDEN.PA", "ENGI.PA", "EL.PA",
    "ERF.PA", "RMS.PA", "KER.PA", "LR.PA", "OR.PA", "MC.PA", "ML.PA", "ORA.PA",
    "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA",
    "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA", "URW.AS",
    "VIE.PA", "DG.PA", "VIV.PA",
]

DAX = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE",
    "BNR.DE", "CBK.DE", "CON.DE", "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE",
    "DPW.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN3.DE",
    "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE",
    "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE",
    "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
]

FTSE = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L", "DGE.L",
    "BATS.L", "GLEN.L", "REL.L", "LSEG.L", "NG.L", "BA.L", "CPG.L", "AAL.L",
    "PRU.L", "TSCO.L", "VOD.L", "LLOY.L", "BARC.L", "NWG.L", "IMB.L", "SSE.L",
    "AHT.L", "RKT.L", "STAN.L", "III.L", "ANTO.L", "WPP.L",
]

SMI = [
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "CFR.SW",
    "LONN.SW", "SIKA.SW", "GIVN.SW", "SREN.SW", "HOLN.SW", "SCMN.SW",
    "GEBN.SW", "SLHN.SW", "ALC.SW", "PGHN.SW", "SOON.SW", "LOGN.SW", "KNIN.SW",
]

AEX = [
    "ASML.AS", "PRX.AS", "INGA.AS", "ADYEN.AS", "AD.AS", "PHIA.AS", "HEIA.AS",
    "WKL.AS", "AKZA.AS", "DSFIR.AS", "KPN.AS", "NN.AS", "ASM.AS", "BESI.AS",
    "RAND.AS", "IMCD.AS", "AGN.AS", "ABN.AS",
]

SP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ", "WMT",
    "NFLX", "ABBV", "CRM", "BAC", "ORCL", "CVX", "MRK", "KO", "AMD", "PEP",
    "TMO", "LIN", "ADBE", "CSCO", "ACN", "MCD", "ABT", "PM", "DHR", "TXN",
    "GE", "INTU", "VZ", "IBM", "QCOM", "CAT", "NEE", "DIS", "CMCSA", "AMGN",
    "PFE", "NOW", "RTX", "UNP", "SPGI", "HON", "LOW", "UPS", "COP", "BKNG",
    "GS", "MS", "BLK", "AXP", "T", "SCHW", "PLD", "SBUX", "MDT", "GILD",
    "DE", "LMT", "ADP", "BA", "MMM", "CVS", "MO", "CI", "SO", "DUK", "TGT",
    "USB", "PNC", "COF", "EMR", "NKE", "F", "GM", "PYPL", "ORLY", "MDLZ",
    "CL", "KHC", "EXC", "AIG", "MET", "SPG", "WBA", "FDX", "CHTR",
]

NASDAQ_TECH = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "MU", "QCOM", "TXN", "AMAT",
    "LRCX", "KLAC", "ADI", "NXPI", "MRVL", "SNPS", "CDNS", "PANW", "CRWD",
    "FTNT", "DDOG", "SNOW", "MDB", "ZS", "NET", "SHOP", "SQ", "ABNB", "UBER",
    "LYFT", "RBLX", "PLTR", "COIN", "HOOD", "SPOT", "PINS", "SNAP", "ZM",
]

NIKKEI = [
    "7203.T", "6758.T", "6861.T", "8306.T", "9432.T", "9984.T", "6098.T",
    "8035.T", "4063.T", "6501.T", "7267.T", "6902.T", "8058.T", "8031.T",
    "4502.T", "6367.T", "7741.T", "9433.T", "8766.T", "6702.T", "4661.T",
    "6273.T", "6954.T", "7974.T", "9983.T",
]

ASIE = [
    "0700.HK", "9988.HK", "3690.HK", "1299.HK", "0005.HK", "0939.HK",
    "2318.HK", "1810.HK", "TSM", "005930.KS", "000660.KS", "D05.SI",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
]

EMERGENTS = [
    "VALE", "PBR", "ITUB", "BBD", "ABEV", "AMX", "WALMEX.MX", "FEMSA.MX",
    "NPN.JO", "SOL.JO", "TUPRS.IS",
]

ETF_MONDE = [
    "IWDA.AS", "EUNL.DE", "CW8.PA", "VWCE.DE", "IWDA.L", "SWDA.L",
    "EIMI.L", "AGGH.MI", "IEMA.AS", "CSPX.L", "VUSA.L", "SXR8.DE",
]

ETF_SECTORIELS = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "XLC", "SMH", "XBI", "ITA", "GDX",
]

ETF_OBLIGATAIRE = [
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB",
    "IEAC.L", "IEGA.AS",
]

MATIERES_PREMIERES = [
    "GC=F", "SI=F", "PL=F", "HG=F", "CL=F", "BZ=F", "NG=F", "ZC=F", "ZW=F",
    "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F",
]

CRYPTO = [
    "BTC-EUR", "ETH-EUR", "SOL-EUR", "BNB-EUR", "XRP-EUR", "ADA-EUR",
    "AVAX-EUR", "DOT-EUR", "LINK-EUR", "MATIC-EUR", "LTC-EUR", "ATOM-EUR",
]

DEVISES = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "EURGBP=X", "EURCHF=X",
    "EURJPY=X", "AUDUSD=X", "USDCAD=X", "USDCNY=X", "USDBRL=X", "USDINR=X",
]

INDICES = {
    "CAC 40": "^FCHI",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "Euro Stoxx 50": "^STOXX50E",
    "Stoxx Europe 600": "^STOXX",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "SMI": "^SSMI",
    "AEX": "^AEX",
    "IBEX 35": "^IBEX",
    "FTSE MIB": "FTSEMIB.MI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "KOSPI": "^KS11",
    "Sensex": "^BSESN",
    "ASX 200": "^AXJO",
    "Bovespa": "^BVSP",
    "TSX": "^GSPTSE",
    "MSCI World (ETF IWDA)": "IWDA.AS",
    "MSCI Émergents (ETF EIMI)": "EIMI.L",
    "VIX (volatilité)": "^VIX",
}

UNIVERS = {
    "CAC 40 (France)": CAC40,
    "DAX 40 (Allemagne)": DAX,
    "FTSE (Royaume-Uni)": FTSE,
    "SMI (Suisse)": SMI,
    "AEX (Pays-Bas)": AEX,
    "S&P 100 (États-Unis)": SP100,
    "Technologie US": NASDAQ_TECH,
    "Nikkei (Japon)": NIKKEI,
    "Asie": ASIE,
    "Émergents": EMERGENTS,
    "ETF actions monde": ETF_MONDE,
    "ETF sectoriels US": ETF_SECTORIELS,
    "ETF obligataires": ETF_OBLIGATAIRE,
    "Matières premières": MATIERES_PREMIERES,
    "Crypto-monnaies": CRYPTO,
    "Devises": DEVISES,
}


def europe() -> list[str]:
    """Grandes capitalisations europeennes, toutes places confondues."""
    return sorted(set(CAC40 + DAX + FTSE + SMI + AEX))


def monde() -> list[str]:
    """Univers large multi-zones. Attention au temps de telechargement."""
    return sorted(set(europe() + SP100 + NIKKEI + ASIE + EMERGENTS))


def taille_univers() -> dict[str, int]:
    return {nom: len(liste) for nom, liste in UNIVERS.items()}

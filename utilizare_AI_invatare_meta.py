
from AI_invatare_meta import DateExperimentale, MetaInvatare

# Încarcă datele
print("Se încarcă datele experimentale...")
date_experimentale = DateExperimentale("tabel_completat.xlsx")
date_experimentale.preprocesare_date()
date_experimentale.genereaza_sarcini()

# Inițializează modelul
print("Se inițializează modelul...")
input_dim = date_experimentale.sarcini[0][0].shape[1]
meta_model = MetaInvatare(alpha=0.1, beta=0.01)
meta_model.initializare_model(input_dim=input_dim, output_dim=4)

# Antrenare
print("Se antrenează modelul...")
meta_model.antrenare(date_experimentale.sarcini, num_iteratii=3000)

# Predicție
print("Se pregătește sarcina pentru predicție...")
selectie = {
    "provenineta_membrana": "Commercial non-impregnated LDPE",
    "solutie_electromigrare_pol_electrod_anod": "NaOH",
    "solutie_electromigrare_pol_electrod_cathode": "HCl",
    "tub_transfer_crown_ether": "18-crown-6",
    "concentratie_crown_ether": 0.05,
    "solutie_organica_+eter_coroana": "Dodecan",
    "setari_aparat": 1.0,
    "potential_electromigrare": 1.2,
}
date_noi = date_experimentale.obtine_date_noi(selectie)
print("Date pentru predicție:", date_noi)

print("Se efectuează predicția...")
pred = meta_model.prezicere(date_noi)
print("Predicții pentru sarcina nouă:", pred)

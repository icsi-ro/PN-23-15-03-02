from completare_date import CompletareDate

# Încarcă fișierul Excel și aplică completarea datelor
completare = CompletareDate('tabel_experiment.xlsx')
completare.preprocesare_coloane()
completare.identifica_coloane_numerice()
completare.aplica_knn()
completare.scala_temporala()
completare.ajusteaza_izotopi()
completare.salveaza_in_excel('tabel_completat.xlsx')

print("Datele lipsă au fost completate și salvate cu succes!")

#################################
###   Investment Optimizer    ###
#################################
### Autor: Florian Wohlkinger ###
### Stand: 08.06.2024         ###
### Version: 1.0.0            ###
#################################


# Module importieren
import pandas as pd                             # für Data Frame
import numpy as np                              # für normalverteilte Zufallsvariable
import matplotlib.pyplot as plt                 # für Visualisierung
from matplotlib.ticker import MultipleLocator   # für Visualisierung (--> Achsenbeschriftung)


# ----------------------------------------------- Funktion Anlagerechner -----------------------------------------------

def anlagerechner(ersteinzahlung, laufzeit, sparrate, dynamik, zinssatz, zins_volatilitaet, intervall, verwaltungskosten_var, verwaltungskosten_fix, abschlusskosten_var, abschlusskosten_fix):
    data = []                                   # Datensatz anlegen
    gesamteinzahlung = ersteinzahlung           # speichert die im Zeitverlauf getätigten Einzahlungen
    gesamtzins = 0                              # speichert die im Zeitverlauf verdienten Zinsen
    gesamtverwaltungskosten = 0                 # speichert die im Zeitverlauf entstandenen Verwaltungskosten

    # Zeile 0 vor Beginn der Laufzeit ("Monat 0") --> Mit der Ersteinzahlung beginnt die Laufzeit
    data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, ersteinzahlung, 0])

    # Schleife für das Kalenderjahr
    for jahr in range(1, laufzeit + 1):
        jahreszins = 0                          # Jahreszins für jedes Kalenderjahr
        kontostand = ersteinzahlung             # Aktueller Kontostand als Grundlage für die Zinsberechnung
        zinssatz_monat_vorher = zinssatz        # Initialisierung des Zinssatzes für den ersten Monat

        # Schleife für die einzelnen Monate
        for monat in range(1, 13):
            zinssatz_monat = zinssatz           # Hilfsvariable (um die originale Eingabe nicht zu überschreiben)

            ### Vorgehensweise bei volatilen Zinsen: Generieren einer normalverteilten Zufallsvariable
            #   mit Abhängigkeit vom Zinssatz im Vormonat --> autoregressive Komponente (AR)
            zins_stddev = zinssatz * zins_volatilitaet / 100
            zufallsvariable = np.random.normal(0, zins_stddev)
            AR_parameter = 0.5      # AR-Parameter für den AR(1)-Prozess

            if zins_stddev == 0:    # Festzins (keine Volatilität)
                zinssatz_monat = zinssatz
            else:                   # Variabler Zinssatz --> AR-Modell
                zinssatz_monat = zinssatz + AR_parameter * (zinssatz_monat_vorher - zinssatz) + zufallsvariable
            zinssatz_monat_vorher = zinssatz_monat                  # Update des Zinssatzes für den nächsten Monat

            # Berechnung der Verwaltungskosten und der monatlichen Investitionsrate
            verwaltungskosten_prozentual = sparrate * verwaltungskosten_var / 100
            verwaltungskosten = verwaltungskosten_prozentual + verwaltungskosten_fix
            investitionsrate = sparrate - verwaltungskosten

            monatsanfang = kontostand                               # Monatsanfang entspricht dem letzten Kontostand
            kontostand += investitionsrate                          # Ergänzen der monatlichen Investitionsrate
            monatszins = kontostand * ((zinssatz_monat / 100) / 12) # Berechnung des im Monat verdienten Zinsbetrags

            jahreszins += monatszins                                # Aktualisierung Jahreszins
            gesamtzins += monatszins                                # Aktualisierung Gesamtzins
            gesamteinzahlung += sparrate                            # Aktualisierung Gesamteinzahlung
            gesamtverwaltungskosten += verwaltungskosten            # Aktualisierung Gesamtverwaltungskosten
            data.append([jahr, monat, zinssatz_monat, round(monatsanfang, 2), sparrate, verwaltungskosten,
                         gesamtverwaltungskosten, investitionsrate, round(monatszins, 2), round(kontostand, 2), 0])

            # Zinsauszahlung je nach Intervall
            if intervall == 1:      # jährlich
                if monat == 12:                               # Zinsauszahlung am Jahresende
                    data[-1][-1] = round(jahreszins, 2)
                    ersteinzahlung = kontostand + data[-1][-1]
            elif intervall == 2:    # quartalsweise
                if monat % 3 == 0:                            # Zinsauszahlung am Ende jedes Quartals
                    data[-1][-1] = round(jahreszins, 2)
                    ersteinzahlung = kontostand + data[-1][-1]
                    jahreszins = 0                            # Reset Jahreszins (="Quartalszins") nach Auszahlung
                    kontostand = ersteinzahlung              # Kontostand wird auf den Stand nach Zinsauszahlung gesetzt
            elif intervall == 3:    # monatlich
                data[-1][-1] = round(monatszins, 2)
                ersteinzahlung = kontostand + data[-1][-1]   # Die Kontostandaktualisierung erfolgt monatlich
                kontostand = ersteinzahlung                  # Kontostand wird auf den Stand nach Zinsauszahlung gesetzt

        # Ggf. jährliche Erhöhung der Sparrate um den Prozentsatz der Dynamik
        if dynamik > 0 and jahr < laufzeit:
            dynamik_betrag = round(sparrate * (dynamik / 100), 2)
            sparrate = round(sparrate + dynamik_betrag, 2)

    # Letzte Zeile am Ende des Datensatzes hinzufügen --> beinhaltet den finalen Kontostand inkl. letzter Zinsauszahlung
    endvermoegen = kontostand + data[-1][-1]                                        # Endvermögen vor Abzug der Abschlusskosten
    abschlusskosten_variabel = round(endvermoegen * abschlusskosten_var / 100, 2)   # Berechnung der variablen Abschlusskosten
    abschlusskosten = abschlusskosten_variabel + abschlusskosten_fix                # Summe beider Abschlusskostenarten
    endvermoegen_nach_abschlusskosten = endvermoegen - abschlusskosten              # Endvermögen nach Abzug der Abschlusskosten
    data.append([laufzeit + 1, 0, 0, 0, 0, abschlusskosten, gesamtverwaltungskosten+abschlusskosten, 0, 0, endvermoegen_nach_abschlusskosten, gesamtzins])

    # Datensatz erstellen
    columns = ["Jahr", "Monat", "Zinssatz", "Monatsanfang", "Monatliche Sparrate", "Kosten", "Kostensumme",
               "Monatliche Investitionsrate", "Monatszins", "Monatsende", "Zinsauszahlung"]
    df = pd.DataFrame(data, columns=columns)

    return df, round(gesamtzins, 2), endvermoegen_nach_abschlusskosten, gesamteinzahlung, gesamtverwaltungskosten, abschlusskosten


# --------------------------------------------------- Programmaufruf ---------------------------------------------------

### Eingabe der Parameter und Funktionsaufruf
def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format

    print("\033[94m========================\033[0m")
    print("\033[94m| Investment Optimizer |\033[0m")
    print("\033[94m========================\033[0m")

    print("\nDieses Programm berechnet die Entwicklung einer Geldanlage über einen bestimmten Zeitraum unter Berücksichtigung verschiedener Parameter.")
    print("    \033[32mAnlageparameter\033[0m: Anlagesumme, Laufzeit, Sparrate, Dynamik (jährliche Erhöhung der Sparrate), Zinssatz, Intervall der Zinsauszahlung")
    print("    \033[31mKostenparameter\033[0m: variable Verwaltungskosten (% der Sparrate), fixe Verwaltungskosten (von der Sparrate abgezogener Festbetrag)")
    print("                     fixe Abschlusskosten (% des Endvermögens), fixe Abschlusskosten (vom Endvermögen abgezogener Festbetrag)")
    print("\nNeben festverzinslichen Geldanlagen können auch volatile Erträge (wie etwa bei Aktienfonds oder ETFs) simuliert werden.")
    print("Die Ergebnisse werden sowohl tabellarisch als auch grafisch über die gesamte Anlagelaufzeit ausgegeben.")

    # Anlageparameter
    print("\n\033[32mAnlageparameter\033[0m:")
    print("----------------")
    ersteinzahlung = float(input("Anlagesumme (Ersteinzahlung): "))
    laufzeit = int(input("Geplante Laufzeit in Jahren: "))
    sparrate = float(input("Monatliche Sparrate: "))
    dynamik = float(input("Jährliche Erhöhung der Sparrate (in Prozent, '0' für keine Dynamik): "))
    zinstyp = int(input("Fester Zinssatz oder variabler Ertrag? '1'=fest, '2'=variabel: "))
    zinssatz = float(input("Zinssatz (p.a.): ")) if zinstyp == 1 else float(
        input("Angenommener durchschnittlicher Zinssatz (p.a.): "))
    zins_volatilitaet = float(input("Volatilität des Zinssatzes (0-100): ")) if zinstyp == 2 else 0
    intervall = int(input("Intervall der Zinsauszahlung: '1'=jährlich, '2'=quartalsweise, '3'=monatlich: "))

    # Kostenparameter
    print("\n\033[31mKostenparameter\033[0m:")
    print("----------------")
    verwaltungskosten_var = float(input("Variable Verwaltungskosten (in Prozent der Sparrate): "))  # Prozentualer Abzug von den Einzahlungen
    verwaltungskosten_fix = float(input("Fixe Verwaltungskosten (in Euro pro Monat): "))            # Festbetrag --> Abzug von den monatlichen Einzahlungen
    abschlusskosten_var = float(input("Variable Abschlusskosten (in Prozent des Endvermögens): "))  # Prozentualer Abzug vom Endvermögen (vor Steuern)
    abschlusskosten_fix = float(input("Fixe Abschlusskosten (als Festbetrag in Euro): "))           # Festbetrag --> Abzug vom Endvermögen (vor Steuern)

    print(f"\nFür die Berechnung der Verwaltungskosten wird von {verwaltungskosten_var}% des Anlagebetrags sowie zusätzlich einem Fixbetrag von {verwaltungskosten_fix}€ ausgegangen.")
    print(f"Zudem werden variable Abschlusskosten i.H.v. {abschlusskosten_var}% vom Endvermögen und ggf. ein Fixbetrag von {abschlusskosten_fix}€ abgezogen.")
    print(f"Außerdem wird die Kapitalertragssteuer i.H.v. 26.375% (inkl. Solidaritätszuschlag) berücksichtigt.")


    # ------------------------------------------------ Funktionsaufruf ------------------------------------------------

    df, gesamtzins, endvermoegen_nach_abschlusskosten, gesamteinzahlung, gesamtverwaltungskosten, abschlusskosten = anlagerechner(
        ersteinzahlung, laufzeit, sparrate, dynamik, zinssatz, zins_volatilitaet, intervall, verwaltungskosten_var,
        verwaltungskosten_fix, abschlusskosten_var, abschlusskosten_fix
    )


    # ----------------------------------- Entwicklung der Geldanlage (--> DataFrame) -----------------------------------

    print("\nEntwicklung der Geldanlage über die Laufzeit:")
    print("-----------------------------------------------")
    print(df)


    # ------------------------------------------- Ergebnisse der Berechnung -------------------------------------------

    print("\nErgebnisse der Berechnung:")
    print("--------------------------")

    # Vermögensentwicklung
    if dynamik > 0:
        letzte_sparrate = df.iloc[-2]['Monatliche Sparrate']
        print(f"Höhe der letzten Sparrate: \033[32m{letzte_sparrate:.2f}€\033[0m")
    print(f"Summe der Einzahlungen über die Laufzeit: \033[32m{gesamteinzahlung:.2f}€\033[0m")
    print(f"Summe der Zinserträge über die Laufzeit: \033[32m{gesamtzins:.2f}€\033[0m")

    # Erträge vs. Kosten
    print(f"Gesamtverwaltungskosten (variabel+fix): \033[91m{gesamtverwaltungskosten:.2f}€\033[0m")
    print(f"Wert der Geldanlage zum Ende der Laufzeit: {endvermoegen_nach_abschlusskosten+abschlusskosten:.2f}€\033[0m")
    print(f"Abschlusskosten (variabel+fix): \033[91m{abschlusskosten:.2f}€\033[0m")
    print(f"Wert der Geldanlage zum Ende der Laufzeit (nach Abschlusskosten): \033[32m{endvermoegen_nach_abschlusskosten:.2f}€\033[0m")
    print(f"Gesamtkosten: \033[91m{gesamtverwaltungskosten + abschlusskosten:.2f}€\033[0m")

    # Steuerabzug
    steuer = (endvermoegen_nach_abschlusskosten - gesamteinzahlung) / 100 * 26.375
    print(f"Kapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): \033[31m{steuer:.2f}€\033[0m")
    print(f"Endvermögen nach Steuern: \033[94m{endvermoegen_nach_abschlusskosten-steuer:.2f}€\033[0m")

    # Analysekennwerte
    print(f"\nROI (Return on Investment) der Geldanlage: \033[32m{((gesamtzins-gesamtverwaltungskosten-abschlusskosten) / gesamteinzahlung) * 100:.2f}%\033[0m")
    if zinstyp == 2:
        print(f"Durchschnittlicher Zinssatz über die Laufzeit: \033[32m{df['Zinssatz'].iloc[1:-1].mean():.2f}%\033[0m")


    # ------------------------------------------------- Visualisierung -------------------------------------------------

    df = df.iloc[1:].reset_index(drop=True)  # Erste Zeile im Datensatz wird gelöscht und der Index neu indiziert!
    plt.figure(figsize=(10, 6))

    # Linie für Kontostand über die Zeit
    # plt.plot(df.index, df['Monatsende'], label='Kontostand', color='blue')  # glatter Kurvenverlauf
    plt.step(df.index, df['Monatsende'], where='post', label='Vermögensstand')  # plt.step --> "Treppenstufen"

    # Bereich unter der Kontostandlinie einfärben
    # plt.fill_between(df.index, df['Monatsende'], color='blue', alpha=0.3)                 # glatter Kurvenverlauf
    plt.fill_between(df.index, df['Monatsende'], step='post', color='blue', alpha=0.3)  # step='post' -> "Treppenstufen"

    # Zweite Linie für kumulierte Einzahlungen
    kumulierte_einzahlungen = df['Monatliche Sparrate'].cumsum() + ersteinzahlung
    # plt.plot(df.index, kumulierte_einzahlungen, label='Einzahlungen', color='orange', linestyle='--')   # glatt
    plt.step(df.index, kumulierte_einzahlungen, where='post', label='Einzahlungen', linestyle='--')  # Treppe

    # Bereich unter der Linie für kumulierte Einzahlungen einfärben
    # plt.fill_between(df.index, kumulierte_einzahlungen, color='orange', alpha=0.3)                  # glatt
    plt.fill_between(df.index, kumulierte_einzahlungen, step='post', color='orange', alpha=0.3)  # Treppe

    # Dritte Linie für kumulative Kosten über die Zeit
    plt.plot(df.index, df['Kostensumme'], label='Kosten', linestyle='--', color='red')

    # Bereich unter der Linie für kumulative Kosten einfärben
    plt.fill_between(df.index, df['Kostensumme'], step='post', color='red', alpha=0.3)  # Treppe

    # Titel und Achsenbeschriftungen
    plt.title('Entwicklung der Geldanlage über die Zeit')
    plt.ylabel('Betrag (in €)')
    plt.xlim(0, len(df) - 0.5)  # Grenzen der x-Achse anpassen

    # Unterschiedliche Skalierung der x-Achse je nach Laufzeit
    if laufzeit <= 4:
        plt.xlabel('Laufzeit (in Monaten)')
        major_locator = MultipleLocator(3)  # Hauptintervall: jedes Quartal
        minor_locator = MultipleLocator(1)  # Zwischenintervall: jeder Monat
        plt.gca().xaxis.set_major_locator(major_locator)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    elif laufzeit > 4 and laufzeit < 8:
        plt.xlabel('Laufzeit (in Monaten)')
        major_locator = MultipleLocator(12)  # Hauptintervall: alle 12 Monate (1 Jahr)
        minor_locator = MultipleLocator(3)  # Zwischenintervall: alle 3 Monate
        plt.gca().xaxis.set_major_locator(major_locator)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    else:
        plt.xlabel('Laufzeit (in Jahren)')
        plt.xticks(np.arange(0, len(df), 12), labels=np.arange(0, len(df) // 12 + 1))

    # Textfeld für die Legende mit den eingegebenen Parametern der Geldanlage und den Ergebnissen der Berechnung
    anlage_text = (
            r"$\bf{Kennzahlen\ der\ Anlage}$"
            + f"\nErsteinzahlung: {ersteinzahlung:.2f}€"
            + f"\nLaufzeit: {laufzeit} Jahre"
            + f"\nAnf. Sparrate: {sparrate:.2f}€"
            + f"\nDynamik: {dynamik:.2f}%"
            + f"\nVerwaltungskosten (variabel): {verwaltungskosten_var}%"
            + f"\nVerwaltungskosten (fix): {verwaltungskosten_fix}€"
            + f"\nAbschlusskosten (variabel): {abschlusskosten_var}%"
            + f"\nAbschlusskosten (fix): {abschlusskosten_fix}€"
            + f"\nAngenommener Zinssatz (p.a.): {zinssatz}%"
            + f"\nZinsauszahlung: "
    )

    if intervall == 1:
        anlage_text += "jährlich"
    elif intervall == 2:
        anlage_text += "quartalsweise"
    elif intervall == 3:
        anlage_text += "monatlich"

    ergebnisse_text = r"$\bf{Ergebnisse}$"
    if dynamik > 0:
        ergebnisse_text += f"\nLetzte Sparrate: {letzte_sparrate:.2f}€"
    if zinstyp == 2:
        ergebnisse_text += f"\nDurchschnittlicher Zinssatz (p.a.): {round(df['Zinssatz'].mean(), 2)}%"
    ergebnisse_text += (
        f"\nSumme der Einzahlungen: {gesamteinzahlung:.2f}€"
        f"\nGesamter Zinsertrag: {gesamtzins:.2f}€"
        f"\nVerwaltungskosten (variabel+fix): {gesamtverwaltungskosten:.2f}€"
        f"\nSparvermögen zum Ende der Laufzeit: {endvermoegen_nach_abschlusskosten+abschlusskosten:.2f}€"
        f"\nAbschlusskosten (variabel+fix): {abschlusskosten:.2f}€"
        f"\nGesamtkosten: {(gesamtverwaltungskosten + abschlusskosten):.2f}€"
        f"\nEndkontostand nach Abschlusskosten: {endvermoegen_nach_abschlusskosten:.2f}€"
        f"\nKapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): {steuer:.2f}€"
        f"\nEndvermögen nach Steuern: {endvermoegen_nach_abschlusskosten-steuer:.2f}€"
    )

    plt.text(0.02, 0.98, anlage_text + "\n\n" + ergebnisse_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Legende anzeigen
    plt.legend(loc='lower right')

    # Anzeigen der Visualisierung
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------- Programmaufruf ---------------------------------------------------

if __name__ == "__main__":
    main()

# Przewidywanie Cen Biletów Lotniczych - Projekt Machine Learning

## 📋 Informacje o Projekcie

**Tytuł**: Sieci neuronowe i uczenie maszynowe  
**Uczelnia**: Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie  
**Wydział**: Zarządzania  
**Data**: Maj 2025

## 👥 Autorzy

- Jakub Kaźmierczyk
- Aleksander Jasiński  
- Piotr Otręba
- Kacper Łapot
- Rafał Łubkowski

## 📊 Opis Zbioru Danych

### Źródło i Metodologia
- **Źródło**: [momondo.pl](https://momondo.pl) 
- **Metoda pozyskania**: Web scraping z użyciem Pythona
- **Okres zbierania**: Od 01.11.2024
- **Zakres czasowy**: Wyszukiwanie lotów z okresem 31 dni następujących po danym dniu

### Trasy i Linie Lotnicze
**Miasta wylotu**: Warszawa, Kraków  
**Miasta docelowe**: Berlin, Paryż, Londyn, Rzym

**Rodzaje połączeń**: Loty bezpośrednie i z jedną przesiadką

**Linie lotnicze**:
- **Tanie**: Ryanair, Wizz Air, EasyJet
- **Tradycyjne**: LOT, Lufthansa, British Airways, KLM, Air France

### Zmienne w Zbiorze Danych

| Zmienna | Opis |
|---------|------|
| `Extraction Weekday` | Dzień tygodnia pobrania obserwacji |
| `Flight Weekday` | Dzień tygodnia lotu |
| `Departure time` | Godzina odlotu |
| `Arrival time` | Godzina przylotu |
| `dep city` | Miasto wylotu |
| `arr city` | Miasto przylotu |
| `Price` | Cena biletu (w zł) |
| `Cabin bag` | Liczba sztuk bagażu podręcznego |
| `Checked bag` | Liczba sztuk bagażu rejestrowanego |
| `Days to departure` | Liczba dni między pobraniem a wylotem |
| `layover duration` | Czas przesiadki (h) |
| `Airline1` | Linia lotnicza wykonująca pierwszy lot |
| `Airline2` | Linia lotnicza wykonująca drugi lot |

## 🧠 Zastosowane Modele

### 1. Sieci Neuronowe (Regresja)

#### Najlepszy Model
- **Architektura**: 1 warstwa ukryta, 16 neuronów
- **Funkcja aktywacji**: ReLU
- **Learning rate**: 0,01
- **Mechanizm**: Early stopping (50 epok bez poprawy)

**Wyniki**:
- **Train MSE**: 137,784
- **Val MSE**: 134,419
- **Test MSE**: 131,173
- **Test MAE**: 233,84
- **Test MAPE**: 0,26
- **Test R²**: 0,87

#### Klasyfikacja (Low-cost vs Traditional)
- **Najlepsza architektura**: [2, 2, 2] (3 warstwy po 2 neurony)
- **Test accuracy**: 0,794
- **Test F1**: 0,706
- **Funkcja aktywacji**: Sigmoid/Tanh (najlepsze wyniki)

### 2. Random Forest

#### Regresja - Najlepszy Model
- **Liczba drzew**: 40
- **Zmienne na podział**: 12
- **min_samples_split**: 5
- **min_samples_leaf**: 2

**Wyniki**:
- **Test MSE**: 76,851
- **Test MAE**: 170,49
- **Test R²**: 0,929
- **Train time**: 337 sekund

#### Klasyfikacja
- **Wyniki**: Wszystkie konfiguracje osiągały niemal perfekcyjne rezultaty
- **Test accuracy**: ~1,000
- **Test F1**: ~1,000

### 3. k-Najbliższych Sąsiadów (k-NN)

#### Regresja - Optymalna Konfiguracja
- **k**: 5
- **Wagi**: Distance
- **Najważniejsze cechy**: Checked_bag, Days_to_departure, Cabin_bag, Flight_time

**Wyniki**:
- **Test MSE**: 189,606
- **Test MAE**: 244,57
- **Test R²**: 0,82

#### Klasyfikacja
- **k**: 7
- **Wagi**: Distance
- **Test F1**: 0,725

### 4. XGBoost ⭐ **Najlepszy Model**

#### Parametry Finalne
- **n_estimators**: 100
- **learning_rate**: 0,1-0,2
- **max_depth**: 10-14
- **lambda**: 1
- **gamma**: 0,1

**Wyniki Regresji**:
- **Train MSE**: 23,047
- **Val MSE**: 58,411
- **Test MSE**: 54,828
- **Test MAPE**: 0,17
- **Test R²**: 0,95

#### Klasyfikacja
- **Najlepsza konfiguracja**: 300 drzew, depth=4, learning_rate=0,10
- **Test accuracy**: 0,76
- **Test F1**: 0,69

## 📈 Porównanie Wyników

| Model | Test R² | Test MAPE | Test MSE | Test MAE |
|-------|---------|-----------|----------|----------|
| **XGBoost** | **0,95** | **0,17** | **54,828** | **197** |
| Random Forest | 0,929 | 0,24 | 76,851 | 170 |
| Sieci Neuronowe | 0,87 | 0,26 | 131,173 | 234 |
| k-NN | 0,82 | 0,28 | 189,606 | 245 |

## 🔍 Kluczowe Odkrycia

### Najważniejsze Cechy
1. **Checked_bag** - największy wpływ na cenę biletu
2. **Days_to_departure** - istotny wpływ horyzontu rezerwacji  
3. **Num_Layovers** - liczba przesiadek
4. **Cabin_bag** - bagaż podręczny
5. **Flight_time** - czas lotu

### Wnioski z Analizy Cech
- Usunięcie `Ticket_class` powodowało największy spadek jakości modelu
- Zmienne czasowe (`Departure_time`, `Arrival_time`, `Flight_time`) są silnie skorelowane
- `Checked_bag` ma kluczowe znaczenie w klasyfikacji linii lotniczych

## 🛠 Stosowane Technologie

- **Python** - główny język programowania
- **pandas** - manipulacja danymi
- **NumPy** - operacje numeryczne  
- **scikit-learn** - algorytmy uczenia maszynowego
- **XGBoost** - gradient boosting
- **matplotlib/seaborn** - wizualizacje

## 📋 Metodologia Badawcza

### Przygotowanie Danych
- Usunięcie wartości odstających (szczególnie długie przesiadki)
- Usunięcie nielicznych braków danych
- Ujednolicenie wartości w kolumnach
- One-hot encoding dla zmiennych nominalnych

### Walidacja Modeli
- Powtarzanie trenowania (5-10 razy dla każdej konfiguracji)
- Walidacja krzyżowa
- Early stopping dla sieci neuronowych
- Systematyczne testowanie hiperparametrów

### Metryki Oceny
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)  
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **F1-score** (dla klasyfikacji)

## 🎯 Wnioski Biznesowe

Model **XGBoost** osiągnął poziom dokładności przydatny w praktyce:
- **95% wyjaśnionej wariancji**
- **17% średni błąd procentowy**
- Możliwość wykorzystania do wskazówek dotyczących optymalnego czasu zakupu
- Wsparcie analizy trendów cenowych na rynku lotniczym

## 📚 Bibliografia

Projekt bazuje na aktualnych badaniach z zakresu przewidywania cen lotniczych, wykorzystując zaawansowane metody uczenia maszynowego i głębokiej analizy danych.

---

**Uwaga**: Projekt ma charakter edukacyjny i badawczy. Przed podjęciem decyzji o zakupie biletu zawsze weryfikuj aktualne ceny bezpośrednio u przewoźników lotniczych.

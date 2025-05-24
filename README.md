# Aplikacija za prepoznavanje lica u realnom vremenu korištenjem dubokog učenja

## Projektni tim

| Ime i prezime | E-mail adresa (FOI)       | JMBAG      | Github korisničko ime |
| ------------- | ------------------------- | ---------- | --------------------- |
| Niko Crnčec   | ncrncec23@student.foi.hr  | 0016164582 | ncrncec23             |
| Elena Pranjić | epranjic23@student.foi.hr | 0016164967 | epranjic23            |

## Opis domene

Prepoznavanje lica u stvarnom vremenu uključuje prepoznavanje lica u videu ili prijenosima uživo, što može biti korisno u aplikacijama kao što su nadzor i sigurnost. Istraživanja u ovom području usmjerena su na razvoj algoritama za prepoznavanje lica ustvarnom vremenu koji mogu raditi u okruženjima stvarnog svijeta s različitim osvjetljenjem i kutovima kamere. Cilj projekta je razvoj i implementacija tehnika temeljenih na dubokom učenju za prepoznavanje lica u stvarnom vremenu. Siamese mreže predstavljaju specifičnu arhitekturu dubokih neuronskih mreža koja je posebno pogodna za zadatke prepoznavanja i verifikacije lica. One se sastoje od dvije identične podmreže koje dijele iste težine i uče prepoznavati sličnost između dva ulazna uzorka, primjerice dvije slike lica. Umjesto da klasificiraju pojedinačne slike, Siamese mreže uče funkciju uspoređivanja koja mjeri koliko su dvije slike slične ili različite. Ova pristup omogućuje efikasno prepoznavanje novih osoba čak i kada mreža nije ranije vidjela njihove slike tijekom treninga, što je idealno za sustave prepoznavanja lica u stvarnom vremenu s velikim brojem korisnika i varijabilnim uvjetima.

## Specifikacija projekta

**Cilj projekta** <br>
Razviti sustav za prepoznavanje lica u stvarnom vremenu koristeći tehnike dubokog učenja i Siamese mreže, koji može pouzdano prepoznati osobe u video streamu unatoč promjenama osvjetljenja, položaja i izraza lica.

**Ulazni podatci** <br>

<ul>
  <li>Video stream s kamere ili unaprijed snimljeni videozapisi.</li>
  <li>Slike lica za treniranje i verifikaciju modela.</li>
</ul>

**Izlazni podatci**

<ul>
  <li>Identifikacija lica prisutnih u video streamu s oznakom osobe ili informacijom o neprepoznatom licu.</li>
  <li>Moguće je prikazati okvir oko prepoznatog lica i ime osobe u stvarnom vremenu.</li>
</ul>

**Funkcionalnosti aplikacije**

<ul>
  <li>Detekcija lica u video streamu.</li>
  <li>Ekstrakcija značajki lica pomoću dubokih neuronskih mreža.</li>
  <li>Usporedba značajki lica koristeći Siamese mrežu za prepoznavanje i verifikaciju.</li>
  <li>Upravljanje bazom poznatih lica (dodavanje, uklanjanje, ažuriranje).</li>
  <li>Prikaz rezultata u realnom vremenu s vizualnim indikatorima.</li>
</ul>

## Tehnologije i oprema

Aplikacija je razvijena u programskom jeziku Python. Trenirani model dostupan je putem priloženog [linka](https://drive.google.com/drive/folders/1e0pLrdvceyLpMGCxLUkESA8ydUk5sDhB?usp=sharing). Zbog ograničenja lokalnog okruženja, model je treniran u Google Colab okruženju, a uz projekt je na GitHub repozitorij dodana i pripadajuća .ipynb datoteka. Tijekom izrade projekta koristili smo TensorFlow za izgradnju i treniranje modela, OpenCV za obradu videa, te Kivy UI Framework za razvoj korisničkog sučelja. Model je treniran na široko korištenom datasetu za prepoznavanje lica pod nazivom Labelled Faces in the Wild (LFW), koji je standard u ovom području primjene.

## Literatura

1. _Aplikacija je napravljena prema uzoru na ovaj [video](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH) koji je in-depth tutorial o dubokom učenju, računalnom vidu, kreiranju Siamese modela i detekciji osobe u realnom vremenu korištenjem OpenCV._
2. _Osim videa korišteno je i istraživanje o one-shot siamese neuronskim mrežama [Siamese_Neural_Networks.pdf](https://github.com/user-attachments/files/20419437/Siamise_Neural_Networks.pdf)_

# NB 2: Brownian motion

```python
import numpy as np
import matplotlib.pyplot as plt
```

```{warning}
Vergeet niet naar main te gaan, te pullen, een nieuwe branch te maken vanaf main en daarin verder te werken!
```


## Recap en uitbreiding

Je hebt al gezien hoe je een botsend deeltjesmodel maakt. Je startte toen met 1 of 2 deeltjes en kan dit uitbreiden naar ~100 deeltjes. Daarbij is het logisch om gebruik te maken van classes.

Voor het maken van een simulatie van gasmoleculen in een gesloten volume zal je logischerwijs de volgende stappen doorlopen:
1. bepalen van initiële condities
2. definiëren van de deeltjes
3. definiëren van volume met randvoorwaarden
4. simulaties bestaande uit:
    - bepaling van positie van deeltjes
    - controleren op onderlinge botsingen 
    - controleren op botsingen met wanden
    - aanpassen van eigenschappen zoals totaal volume of temperatuur
    - opslaan van gegevens
    
```{note}
Alhoewel het interessant is om steeds een animatie te maken van de beweging van de deeltjes, vraagt dit veel rekentijd. Je kan ook regelmatig een plot maken om te controleren of het programma werkt zoals je verwacht.
```


### De initiële condities

In onze simulatie gaan we uit van een aantal aannames. Bij deze aannames wil je dat de simulatie een voorspellende waarde heeft voor een reëel systeem, maar ook dat de benodigde rekenkracht voor het uitvoeren van de simulatie op te brengen is. We kiezen daarom voor de volgende condities:

- Onze model heeft een beperkt en constant aantal deeltjes.
- We maken gebruik van een 2D simulatie.
- De deeltjes voelen geen onderlinge kracht en ondergaan alleen elastische botsingen.
- (tijdens simuleren) De tijdstap is voldoende klein. (d.w.z. de afgelegde weg $v_0*dt$ is klein in vergelijking met de diameter van de deeltjes, zodat we geen botsingen missen)

Hieronder geven we een aantal constanten voor onze code. Om de simulatie straks te kunnen interpreteren, vergelijken we deze eerst met een realistische situatie:

```{exercise} Lucht in een fietsband
:label: ex-recap-1

Als model gaan we uit van een deel van het volume van een fietsband die onder een druk van 10 bar staat. Laat zien dat het gemiddelde volume $V_{molecuul}$ ongeveer overeenkomt met een kubus met ribben van $1.6 \; \mathrm{nm}$ is. Het antwoord gebruik je zo direct in de code hieronder. 
```

```{solution} ex-recap-1
Met de ideale gaswet $pV = Nk_BT$ is het volume per molecuul

$$
V_\text{molecuul} = \frac{k_B T}{p} = \frac{1{,}38\cdot10^{-23}\ \mathrm{J/K} \cdot 293\ \mathrm{K}}{10\cdot10^{5}\ \mathrm{Pa}} = 4{,}0\cdot10^{-27}\ \mathrm{m^3}
$$

$$
V_\text{molecuul}^{1/3} = \left(4{,}0\cdot10^{-27}\right)^{1/3} = 1{,}6\ \mathrm{nm}
$$
```

De simulatie die we gaan maken bevat maar twee dimensies. We kunnen het oppervlak per deeltje in deze simulatie daarom kiezen als het zojuist berekende $V_\text{molecuul}^{2/3}$.

Voor de gemiddelde snelheid, $v$, van een molecuul geldt dat

$$
    \frac{1}{2}m\left< v^2\right> = \frac{f}{2} k_B T.
$$

Hierbij is $m$ de massa van het deeltje, $f$ het aantal dimensies, $k_B$ de constante van Boltzmann en $T$ de temperatuur.

```{exercise} Snelheid van een gasmolecuul
:label: ex-recap-2
Laat zien dat de snelheid van een gasmolecuul bij kamertemperatuur in twee dimensies ongeveer $4\cdot 10^2 \, \mathrm{m/s}$ is.
Ook deze heb je hierna nodig.
```

```{solution} ex-recap-2
De lucht bestaat vooral uit N$_2$ en O$_2$. Dus kan de gemiddelde massa berekend worden met $m \approx 29\ \mathrm{u} = 29 \cdot 1{,}66\cdot10^{-27} = 4{,}8\cdot10^{-26}\ \mathrm{kg}$.

In 2D hebben we $f = 2$, dus $\tfrac12 m\langle v^2\rangle = k_BT$ en

$$
v = \sqrt{\frac{2 k_B T}{m}} = \sqrt{\frac{2 \cdot 1{,}38\cdot10^{-23} \cdot 293 }{4{,}8\cdot10^{-26}}} = 4{,}1\cdot10^{2}\ \mathrm{m/s}
$$
```

```{exercise} Radius
:label: ex-recap-3
In de python cell hieronder staan enkele waarden gegeven. Het aantal deeltjes $N=40$ is beperkt, dit om de rekentijd van het model te beperken. 
```

```{exercise} DT
:label: ex-recap-4
Bekijk de waarde van `DT`. Hoe zorgt deze waaarde ervoor dat het onwaarschijnlijk is dat een botsing gemist wordt? Zet in het groene commentaar bij elke constante in welk eenheid deze wordt weergegeven.
```

```{solution} ex-recap-4
Per tijdstap legt een deeltje een afstand $v_0 \cdot DT$ af. Voor $v_0 \approx 4\cdot10^2\ \mathrm{m/s}$ en $DT = 10^{-13}\ \mathrm{s}$ is dat $4\cdot10^{-11}\ \mathrm{m} = 0{,}04\ \mathrm{nm}$. Dat is maar ongeveer 10% van de diameter van een deeltje. De overlap tussen de deeltjes zal worden gezien en bovendien kunnen ze niet door elkaar springen.
```

Onze simulatie modelleert dus een klein systeem zonder rekening te houden met de kracht tussen de luchtmoleculen. Je zult zien dat we toch al behoorlijk wat van de processor vragen om deze berekeningen te doen. Om een meer realistische modelsimulatie te maken is daarom een serieuze uitdaging, die veel wiskunde en programmeerervaring vraagt. Op dit ogenblik wordt dit bijvoorbeeld gevraagd bij een van de eerste opdrachten van het vak 'Computational Physics' van de master Applied Physics.

```python tags=["NB2_init"] editable=true slideshow={"slide_type": ""}
BOX_SIZE_0 = 10                # Hoogte en lengte startvolume [nm]
N = 40                         # Aantal deeltjes [-]
V_0 = 1                        # Startsnelheid van deeltjes [v_0 ≈ 4e2 m/s]
RADIUS = 0.3                   # Straal van moleculen [nm]
DT = 0.1 * RADIUS / V_0        # Tijdstap om geen botsing te missen [nm/v_0 ≈ 2,5 ps]
```

### Particle class
Ons `ParticleClass` definieert van een deeltje de massa $m$, snelheid $v$, positie $r$, en straal $R$. De positie moet per tijdstap bepaald worden. 

Nieuw aan `ParticleClass` zijn twee properties, nl. `momentum` en `kin_energy`. Een property is eigenlijk een variabele die samenhangt met de waarde van andere variabelen binnen de klasse. Om de interne consistentie te bewaren moet je de juiste waarde van deze variabele dus telkens afleiden en dat doe je met een functie. Zo geeft `kin_energy` de kinetische energie terug van het deeltje door die af te leiden van de variabelen `m` en `v`.

Buiten `ParticleClass` voegen we twee functies toe, die je ook al hebt gezien. De eerste heet `collide_detection` en detecteert of twee deeltjes onderling overlappen en dus botsen. De tweede heet `particle_collision` en berekent de nieuwe snelheden van de twee deeltjes ten gevolge van hun botsing.

```python editable=true slideshow={"slide_type": ""} tags=["NB2_init"]
# Maken van de class
class ParticleClass:
    def __init__(self, m, v, r, R):
        """ maakt een deeltje (constructor) """
        self.m = m                         
        self.v = np.array(v, dtype=float)  
        self.r = np.array(r, dtype=float)  
        self.R = R

    def update_position(self):
        """ verandert positie voor één tijdstap """
        self.r += self.v * DT 
            
    @property
    def momentum(self):
        return self.m * self.v
    
    @property
    def kin_energy(self):
        return 1/2 * self.m * np.dot(self.v, self.v)
    
def collide_detection(p1: ParticleClass, p2: ParticleClass) -> bool:
    """ Geeft TRUE als de deeltjes overlappen """
    dx = p1.r[0]-p2.r[0]
    dy = p1.r[1]-p2.r[1]
    rr = p1.R + p2.R
    return  dx**2+dy**2 < rr**2 
    #Note: np.linalg.norm(p1.r - p2.r) < (p1.R + p2.R) had ook gekund, maar is veel langzamer. Controleer zelf!

def particle_collision(p1: ParticleClass, p2: ParticleClass):
    """ past snelheden aan uitgaande van overlap """
    m1, m2 = p1.m, p2.m
    delta_r = p1.r - p2.r
    delta_v = p1.v - p2.v
    dot_product = np.dot(delta_r, delta_v)

    # Als deeltjes van elkaar weg bewegen dan geen botsing
    if dot_product >= 0: # '='-teken voorkomt ook problemen als delta_r == \vec{0}
        return
        
    distance_squared = np.dot(delta_r, delta_r) 
    # Botsing oplossen volgens elastische botsing in 2D
    p1.v -= 2 * m2 / (m1 + m2) * dot_product / distance_squared * delta_r
    p2.v += 2 * m1 / (m1 + m2) * dot_product / distance_squared * delta_r
```

### Volume en randvoorwaarden

Voor het volume maken we gebruik van het oppervlak van een `pyplot`, net zoals we dat in de vorige werkbladen gedaan hebben. Daar is op dit ogenblik niets anders voor nodig dan een functie die de deeltjes laat botsen met de wanden.

Je hebt hiervoor al een voorbeeld van gezien, maar hieronder gebruiken we een andere vorm. Als je code ontwikkelt is het zaak om regelmatig en structureel te controleren of je code klopt. Het maakt daarbij niet uit of de code door jezelf is gemaakt, door een ander (zoals de docent) is aangeboden of van een generatieve AI komt. Hoe langer je code toevoegt zonder een controle uit te voeren hoe groter de kans is dat je fout op fout stapelt en niet meer kan traceren wat er precies mis is. 

We maken gebruik van een truc die in veel simulaties wordt gebruikt: we maken gebruik van periodieke randvoorwaarden. In dit geval wil dit zeggen dat een deeltje dat botst met de wand aan de ene kant van het volume, aan de andere kant verschijnt met exact dezelfde snelheid. Op deze manier verandert de snelheid van een deeltje niet bij het botsen met de wand en blijven alle behoudswetten gelden. Die behoudswetten kunnen we straks goed controleren.

```python editable=true slideshow={"slide_type": ""} tags=["NB2_init"]
def box_collision(particle: ParticleClass):
    ''' botsing met wanden geeft periodieke randvoorwaarden '''
    for i in range(2):
        if particle.r[i] > BOX_SIZE_0 / 2:
            particle.r[i] -= BOX_SIZE_0 
        elif particle.r[i] < -BOX_SIZE_0 / 2:
            particle.r[i] += BOX_SIZE_0
```

### Maken van de simulatie

Nu dat we alle benodigde functies hebben, kunnen we ze in de juiste structuur zetten om de simulatie uit te voeren.


#### Bepaling positie deeltjes

Eerst moeten we alle deeltjes aanmaken en de startwaardes van hun variabelen invoeren:


```{exercise}
:label: ex-recap-5
Leg uit dat d.m.v. onderstaande code de deeltjes allemaal dezelfde snelheid hebben en uniform verdeeld zijn over de box.
```

```{solution} ex-recap-5
$v_x$ wordt willekeurig gekozen tussen $-v_0$ en $v_0$, en daarna $v_y = \pm\sqrt{v_0^2 - v_x^2}$. Daardoor is $|\vec v| = v_0$ voor elk deeltje. Het enige wat kan verschillen is de richting van de snelheid.

De $x$- en $y$-coördinaat worden getrokken met `np.random.uniform`, waarbij elke positie even waarschijnlijk is binnen de box.
```

```python tags=["NB2_init"] editable=true slideshow={"slide_type": ""}
def create_particles(particles):
    ''' Leegmaken en opnieuw aanmaken van deeltjes '''
    particles.clear
    for i in range(N):
        vx = np.random.uniform(-V_0, V_0)
        vy = np.random.choice([-1, 1]) * np.sqrt(V_0**2 - vx**2)        
        pos = np.random.uniform(-BOX_SIZE_0/2 + RADIUS, BOX_SIZE_0/2 - RADIUS, 2)
        particles.append(ParticleClass(m=1.0, v=[vx, vy], r=pos, R=RADIUS))
```

#### Controleren op onderlinge botsingen

Om de botsingen van alle deeltjes goed te verwerken moeten we controleren welke paren deeltjes met elkaar moeten botsen en deze botsingen één keer uitvoeren. 


```{exercise}
:label: ex-recap-6
Leg uit hoe er in onderstaande code wordt voorkomen dat de botsing tussen twee deeltjes dubbel wordt uitgevoerd.
```

```{solution} ex-recap-6
De binnenste lus begint bij j = i + 1 in plaats van bij 0. Hierdoor wordt elk paar van i en j een keer gecontroleerd, waarbij altijd geldt dat $j > i$. Het paar $(j, i)$ komt nooit aan bod, omdat deeltje $i$ al met deeltje $j$ is vergeleken toen de buitenste lus bij $i$ was. Deeltjes worden ook niet met zichzelf vergeleken omdat j niet gelijk mag zijn aan i.
```

```python editable=true slideshow={"slide_type": ""} tags=["NB2_init"]
def handle_collisions(particles):
    """ alle onderlinge botsingen afhandelen voor deeltjes in lijst """
    num_particles = len(particles)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if collide_detection(particles[i], particles[j]):
                particle_collision(particles[i], particles[j])
```

#### Controleren op botsingen met wanden

Voor het bepalen of deeltjes botsen met de wanden maken we gebruik van de functie die we net hebben aangepast.

```python editable=true slideshow={"slide_type": ""} tags=["NB2_init"]
def handle_walls(particles):
    ''' botsing met wanden controleren voor alle deeltjes in volume '''
    for p in particles:
        box_collision(p)
```

#### Integreren in tijdstap

Nu moeten we alle functies op de juiste manier combineren om de simulatie een tijdstap te laten maken:

```python editable=true slideshow={"slide_type": ""} tags=["NB2_init"]
def take_time_step(particles):
    for p in particles:
        p.update_position()
        
    handle_collisions(particles)
    handle_walls(particles)  

```

### Uitvoeren simulatie en tonen resultaat

Zoals aangegeven, kunnen we een animatie maken van de positie en snelheid als functie van de tijd, maar we kunnen ook het eindresultaat tonen en interpreteren:

```python
particles = []
create_particles(particles)
for i in range(100):
    take_time_step(particles)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.xlim(-BOX_SIZE_0/2, BOX_SIZE_0/2)
plt.ylim(-BOX_SIZE_0/2, BOX_SIZE_0/2)

for p in particles:
    plt.plot(p.r[0], p.r[1], 'k.', ms=25)
    plt.arrow(p.r[0], p.r[1], p.v[0], p.v[1], 
              head_width=0.05, head_length=0.1, color='red')
plt.show()
```

### Toevoegen animatie

Zoals gezegd: het kost veel meer rekenkracht om een animatie te maken over de tijd. Soms kan het wel helpen om te zien of de simulatie aan alle verwachtingen voldoet. 

```python
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-BOX_SIZE_0/2, BOX_SIZE_0/2)
ax.set_ylim(-BOX_SIZE_0/2, BOX_SIZE_0/2)
ax.set_aspect('equal')
dot, = ax.plot([], [], 'ro', ms=13);

def init():
    dot.set_data([], [])
    return dot,

def update(frame):
    take_time_step(particles)
    dot.set_data([p.r[0] for p in particles], [p.r[1] for p in particles])
    return dot,

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)
HTML(ani.to_jshtml())
```

### Controle code

Om te controleren of onze code werkt passen we nu de code aan.

```{exercise}
:label: ex-recap-7
Voeg aan de simulatie een regel toe die voor elke tijdstap bepaalt wat de totale kinetische energie is voor alle deeltjes en neem deze waarde op in een array die bij elke tijdstap met één element groeit in lengte. Maak vervolgens een plot van de totale kinetische energie als functie van de tijd gedurende een simulatie van 100 tijdstappen.
```

```python tags=["NB2_ex7"]
# plotten van kinetische energie als functie van tijd

particles = []
energy_tot = []
create_particles(particles)

for i in range(100):
    take_time_step(particles)
    E_kin = sum(0.5 * p.m * np.dot(p.v, p.v) for p in particles)
    energy_tot.append(E_kin)

time = np.arange(1, len(energy_tot) + 1) * DT

plt.figure()
plt.plot(time, energy_tot, 'b.-')
plt.xlabel("Time (s)")
plt.ylabel(f"E_total (J)")
plt.ylim(0, 1.2 * max(energy_tot))
plt.show()

# Om de afwijking beter te bestuderen
energy_tot = np.array(energy_tot)
afwijking = energy_tot - energy_tot[0]

# plt.figure()
# plt.plot(time, afwijking, 'r.-')
# plt.xlabel("Time (simulatie-eenheden)")
# plt.ylabel(r"$E - E_0$ (simulatie-eenheden)")
# plt.show()

print("max afwijking:", np.max(np.abs(afwijking)))
print("relatieve afwijking:", np.max(np.abs(afwijking)) / energy_tot[0])
```

```{exercise} 
:label: ex-recap-8
Als je heel nauwkeurig naar de waarde voor de totale kinetische energie kijkt, dan varieert deze toch. Zoom hiervoor goed in op de as voor de energie. Op welke energieschaal gebeurt dit en is de oorzaak?
```

```{solution} ex-recap-8
De totale kinetische energie is $E_0 = N \cdot \tfrac12 m v_0^2 = 40 \cdot \tfrac12 \cdot 1 \cdot 1^2 = 20$. De energie varieert op een schaal van ongeveer $10^{-15}$–$10^{-14}$. Relatief is dat een afwijking van ordegrootte $10^{-16}$–$10^{-15}$.

De oorzaak is niet natuurkundig maar numeriek, namelijk afrondingsfouten. Python rekent met 64-bits floating point getallen, die ongeveer 16 significante cijfers hebben. Bij elke botsing worden de nieuwe snelheden berekend en elke operatie wordt afgerond. Daardoor is de energie na een botsing niet gelijk aan die ervoor.
```


Voor simulaties waarbij de kans op botsingen heel groot is (richting vloeistof), is er een handige methode om de snelheid van je simulatie op te voeren.
In onderstaande code wordt daar een voorbeeld van gegeven.

```{code} python
def handle_collisions(particles):
    ignore_list = []
    for i, p1 in enumerate(particles):
        if p1 in ignore_list:
            continue
        for j, p2 in enumerate(particles):
            if p1 is p2:
                continue                
            if collide_detection(particles[i], particles[j]):
                particle_collision(particles[i], particles[j])
                ignore_list.append(p2)
```

```{exercise} Controleren van beperkingen 🌶 
:label: ex-recap-10

Leg uit wat deze code precies doet. 

Maak een script waarbij twee deeltjes met dezelfde (absolute) snelheid op een derde deeltje afbewegen en tegelijk zouden botsen.
- Laat zien dat 'slechts' twee deeltjes een interactie met elkaar aan gaan.
- Leg uit waarom dat een acceptabele benadering is wanneer je veel deeltjes botsingen per tijdseenheid hebt.
- Voor wie een uitdaging wil: maak een grafiek met het aantal botsingen per tijdseenheid. Dit om bovenstaande benadering verder te verantwoorden.
```

```{solution} ex-recap-10
Voor elk deeltje worden alle andere deeltjes gecontroleerd op een botsing. Na een botsing komt het andere deeltje op de `ignore_list` en wordt het later als `p1` overgeslagen. Zo wordt een botsing niet dubbel uitgevoerd en hoeven er minder paren gecontroleerd te worden, wat de code efficienter maakt.


In het testscript raken A en B tegelijk deeltje C. De code verwerkt dit als twee losse botsingen na elkaar (A–C, dan C–B), dus er botsen steeds maar twee deeltjes. Een tijdstap later klopt het weer ($v_A = -1$, $v_C = 0$, $v_B = +1$).

Driedeeltjesbotsingen binnen een $DT$ zijn zeldzaam. De impuls en energie blijven dan ook behouden. De simulatie wordt hierdoor veel sneller en bij veel botsingen heffen de kleine afwijkingen tegen elkaar op.
```


```python
dt = 0.01
botsingen = []
tijd, xA, xB, xC = [], [], [], []

# Deeltje met een naam, zodat we kunnen zien wie met wie botst
class Particle(ParticleClass):
    def __init__(self, naam, r, v, m=1.0, R=0.45):
        super().__init__(m=m, v=v, r=r, R=R)
        self.naam = naam

# handle_collisions met ignore_list en het bijhouden van botsingen
def handle_collisions(particles):
    ignore_list = []
    for i, p1 in enumerate(particles):
        if p1 in ignore_list:
            continue
        for j, p2 in enumerate(particles):
            if p1 is p2:
                continue
            if collide_detection(particles[i], particles[j]):
                v_voor = p1.v.copy()
                particle_collision(particles[i], particles[j])
                if not np.array_equal(v_voor, p1.v):    # botsing is echt uitgevoerd
                    botsingen.append((stap, p1.naam, p2.naam))
                ignore_list.append(p2)

A = Particle("A", [-2, 0], [ 1, 0])
C = Particle("C", [ 0, 0], [ 0, 0])
B = Particle("B", [ 2, 0], [-1, 0])
particles = [A, C, B]

for stap in range(300):
    for p in particles: p.r += p.v * dt
    handle_collisions(particles)
    tijd.append(stap * dt)
    xA.append(A.r[0]); xB.append(B.r[0]); xC.append(C.r[0])

for i in botsingen:
    print(f"In stap {i[0]}: botsing tussen {i[1]} en {i[2]}")

plt.plot(tijd, xA, 'r', label='A')
plt.plot(tijd, xC, 'g', label='C')
plt.plot(tijd, xB, 'b', label='B')
plt.xlabel("tijd")
plt.ylabel("x-positie")
plt.legend()
plt.show()

print("Eindsnelheden:", {p.naam: float(p.v[0]) for p in particles})
```

````{tip} Nog sneller
Er is een manier om je script nog sneller te maken. 
Die methode gaat wel voorbij aan wat we verwachten van eerstejaars, maar het idee is de moeite waard om te bekijken.
Bij $N$ deeltjes zou je $\frac{N(N-1)}{2}$ mogelijke botsingen moeten bekijken. 
Door gebruik te maken van een ignore list maak je die mogelijk aantal botsingen al een stuk minder.
Maar je zou ook er van uit kunnen gaan dat botsingen slechts plaats vinden met deeltjes die in de buurt zijn.
Als je een grid maakt en elk deeltje in een grid zet, hoeft elk deeltje slechts in de naastgelegen velden te "kijken" of daar een deeltje aanwezig is waarmee een botsing kan plaats vinden.

In onderstaande figuur is dat weergegeven. 
Het lichtrode deeltje heeft potentieel maar negen mogelijke botsingen. 
Dat script kan verder geoptimaliseerd worden door van links naar rechts per grid te lopen, dan hoeven zelfs niet alle velden bekeken te worden.

```{figure} ../figures/grid.png
:width: 50%
```

````


In de volgende notebooks maken we veelal gebruik van numpy functies.
In sommige gevallen is dat sneller, maar lang niet altijd.
Het is good-practice om na te gaan wat sneller is.
Aan de andere kant moet de tekst ook leesbaar blijven...

Als onderdeel om excellent te scoren verwachten we dat je een aantal checks tussendoor doet om code sneller te maken.
In onderstaande testcode, gemaakt door Josh Sleijfer wordt een van die functies getest.
Wat blijkt?
Voor "kleine" vectoren is het sneller om geen gebruik te maken van Numpy maar voor grotere vectoren wel!

```{code} python
import numpy as np
import time

L = 2                # lengte van de vector
repeats = 1_000_000  # aantal herhalingen

rnd_vec1 = np.random.rand(L, repeats)
 
# test numpy dot product
t0 = time.perf_counter()
for vec1 in rnd_vec1.T:
    dot = np.dot(vec1, vec1)
t1 = time.perf_counter()

print(f"Numpy: Time taken for {repeats} dot products of length {L}: {t1 - t0:.4f} seconds")

# test no numpy dot product 
t0 = time.perf_counter()
for vec1 in rnd_vec1.T:
    dot = vec1[0] * vec1[0] + vec1[1] * vec1[1] # + vec1[2] * vec1[2] + vec1[3] * vec1[3] # don't forget to adjust!
t1 = time.perf_counter()

print(f"No Numpy: Time taken for {repeats} dot products of length {L}: {t1 - t0:.4f} seconds")

# test fully vectorized 
t0 = time.perf_counter()
dot = np.sum(rnd_vec1[0] * rnd_vec1[0] + rnd_vec1[1] * rnd_vec1[1])
t1 = time.perf_counter()
print(f"Fully vectorized: Time taken for {repeats} dot products of length {L}: {t1 - t0:.4f} seconds")

```



## Brownian motion

Een van de bekendste voorbeelden van botsende deeltjes in de natuur is Brownian motion.
Fijn gemalen pollen in water lijken te dansen in willekeurige richting.
Dit komt doordat de pollen worden geraakt door watermoleculen die in alle richtingen bewegen.
Omdat de pollen veel zwaarder zijn dan watermoleculen, dus de beweging van de pollen is veel langzamer en minder "intens" dan die van de watermoleculen.
Dit proces van willekeurige beweging door botsingen met kleinere deeltjes wordt Brownian motion genoemd en kunnen we simuleren op basis van ons (premature) botsingsmodel.
Daarbij kunnen we ook gebruik maken van de zojuist geleerde manier van tracking van deeltjes, waarbij we een zowel het zware bolletjes als een enkel deeltje kunnen volgen.

Let op!
We bestuderen hier nog geen thermische effecten, deze opdrachten zijn met name bedoeld om beter te begrijpen hoe het botsingsmodel in elkaar zit.

Voor een theoretische achtergrond over Brownian motion, verwijzen we naar de [Feynman lectures on Physics](https://www.feynmanlectures.caltech.edu/I_41.html).

```{warning}
In dit notebook zitten delen waar ruimte is om code toe te voegen, maar waarbij je denkt... waarom dan?
In een latere opdracht moet je terug naar die cell en de juiste code toevoegen.
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python tags=["NB2_init"]
# Maken van de ParticleClass

class ParticleClass:
    # Het maken van het deeltje
    def __init__(self, m, v, r, R, c):
        self.m = m                         
        self.v = np.array(v, dtype=float)  
        self.r = np.array(r, dtype=float)  
        self.R = R  
        self.c = c

    # Het updaten van de positie, eventueel met zwaartekracht
    def update_position(self):
        self.r += self.v * dt #+ 1/2 * a * dt**2  
              
    # Harde wand
    def boxcollision(self):
        if abs(self.r[0]) + self.R > Box_length: 
            self.v[0] = -self.v[0]                                  # Omdraaien van de snelheid
            self.r[0] = np.sign(self.r[0]) * (Box_length - self.R)  # Zet terug net binnen box                 
        if abs(self.r[1]) + self.R > Box_length: 
            self.v[1] = -self.v[1]     
            self.r[1] = np.sign(self.r[1]) * (Box_length - self.R) 
            
    @property
    def momentum(self):
        return self.m * self.v
    
    @property
    def kin_energy(self):
        return 1/2 * self.m * np.dot(self.v, self.v)
```

```python tags=["NB2_init"]
# zijde van de doos (simulatie-eenheden)
Box_size_0 = 10
# halve zijde: doos loopt van -5 tot +5
Box_length_0 = Box_size_0/2
# de grootte van de doos kan zich aanpassen
Box_length = Box_length_0


# lijst voor alle deeltjes
particles = []
# totaal aantal deeltjes
N = 40
 # snelheid van de gasdeeltjes
v_0 = 1
# tijdstap
dt = 0.04

# N-1 lichte gasdeeltjes: |v| = v_0, met willekeurige richting en positie
for i in range(N-1):
    vx = np.random.uniform(-v_0, v_0)
    vy = np.random.choice([-1, 1]) * np.sqrt(v_0**2 - vx**2)
    pos = Box_length_0 * np.random.uniform(-1, 1, 2)
    particles.append(ParticleClass(m=1.0, v=[vx, vy], r=pos, R=.5, c='blue'))

# Brownse deeltje: 20x zo zwaar, in rust in het midden, beweegt alleen door botsingen
particles.append(ParticleClass(m=20.0, v=[0, 0], r=[0, 0], R=.5, c='red'))
```

```{exercise} Brownian motion
:label: ex-brownian-1

Leg uit wat er in de laatste regel van bovenstaande script gebeurt.
Voeg voor deze regel goede metadata toe.
```

```{solution} ex-brownian-1
In de laatste regel wordt een extra deeltje aan de lijst `particles` toegevoegd, namelijk het Brownse deeltje. Omdat de lus maar tot N - 1 loopt, zijn er 39 lichte gasdeeltjes en 1 zware, en samen dus weer 40.

Het zware deeltje beweegt alleen doordat de lichte deeltjes er willekeurig tegenaan botsen. Elke botsing geeft een klein, willekeurig duwtje. Omdat het deeltje zwaar is, verandert elke botsing de snelheid maar een klein beetje en beweegt het veel lamgzamer dan het gas.
```


Er is een doos vol met deeltjes op willekeurige positie aangemaakt.
We willen kijken waar de deeltjes zijn terechtgekomen.
Hieronder staat dit weergegeven. 



```python
# Inspecteren van beginsituatie
plt.figure()

plt.xlabel('x')
plt.ylabel('y')

plt.xlim(-Box_length_0,Box_length_0)
plt.ylim(-Box_length_0,Box_length_0)


for particle, particle_object in enumerate(particles):
    plt.plot(particle_object.r[0], particle_object.r[1], color=particle_object.c, marker='.')
    plt.arrow(particle_object.r[0], particle_object.r[1],
              particle_object.v[0], particle_object.v[1],
              head_width=0.05, head_length=0.1, color='red')
plt.show()

```

<!-- #region -->
```{exercise} 
:label: ex-brownian-2

Er staat ook code met comments ervoor, wat doet deze code? 
Check je antwoord door de comments weg te halen.
Hoe wordt er voor gezorgd dat de snelheid van elk deeltje gelijk is?
```

```{solution} ex-brownian-2
**Wat doet de uitgecommentarieerde code?**
`plt.arrow(x, y, dx, dy, ...)` tekent een pijl die begint in het punt $(x, y)$ en eindigt in $(x+dx, y+dy)$. Hier begint de pijl op de positie van het deeltje ($\vec r$) en zijn $dx$ en $dy$ de snelheidscomponenten $v_x$ en $v_y$. Elke pijl is dus de **snelheidsvector** van een deeltje: de richting laat zien waar het deeltje naartoe beweegt, de lengte hoe snel. `head_width` en `head_length` bepalen de grootte van de pijlpunt, en `color='red'` maakt de pijlen rood.

Haal je de `#` weg, dan zie je bij elk blauw gasdeeltje een pijl in een willekeurige richting, allemaal even lang. Bij het rode Brownse deeltje zie je geen pijl, omdat de snelheid $(0, 0)$ is.

**Hoe is de snelheid van elk deeltje gelijk?**
Bij het aanmaken wordt $v_x$ willekeurig gekozen tussen $-v_0$ en $v_0$. Daarna wordt $v_y$ berekend als

$$
v_y = \pm\sqrt{v_0^2 - v_x^2}
$$

Daardoor geldt voor elk gasdeeltje

$$
|\vec v| = \sqrt{v_x^2 + v_y^2} = \sqrt{v_x^2 + v_0^2 - v_x^2} = v_0
$$

Alleen de richting is willekeurig. Het teken van $v_y$ wordt met `np.random.choice([-1, 1])` gekozen, zodat deeltjes ook omlaag kunnen bewegen. De grootte is voor alle gasdeeltjes gelijk aan $v_0 = 1$. Daarom zijn alle pijlen in de figuur even lang.
```

We gaan nu de functies van de simulatie weer aanroepen:

```python
# Het bepalen of er een botsing plaats vindt
def collide_detection(self, other):
    """ True als twee deeltjes overlappen (afstand < som van de stralen) """
    dx = self.r[0] - other.r[0]
    dy = self.r[1] - other.r[1]
    rr = self.R + other.R
    return dx**2 + dy**2 < rr**2

def particle_collision(p1: ParticleClass, p2: ParticleClass):
    """ past snelheden aan uitgaande van overlap """
    m1, m2 = p1.m, p2.m
    delta_r = p1.r - p2.r
    delta_v = p1.v - p2.v
    dot_product = np.dot(delta_r, delta_v)

    # Als deeltjes van elkaar weg bewegen dan geen botsing
    if dot_product >= 0:  # '='-teken voorkomt ook problemen als delta_r == \vec{0}
        return False

    distance_squared = np.dot(delta_r, delta_r)
    # Botsing oplossen volgens elastische botsing in 2D
    p1.v -= 2 * m2 / (m1 + m2) * dot_product / distance_squared * delta_r
    p2.v += 2 * m1 / (m1 + m2) * dot_product / distance_squared * delta_r
    return True

def handle_collisions(particles):
    """ alle onderlinge botsingen afhandelen voor deeltjes in lijst """
    num_collisions = 0
    num_particles = len(particles)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if collide_detection(particles[i], particles[j]):
                if particle_collision(particles[i], particles[j]):
                    num_collisions += 1
    return num_collisions
```

In onderstaande code geven we de code voor de simulatie en volgen we de positie van het zware deeltje. 
<!-- #endregion -->

```python
# De functies van de cel erboven
def collide_detection(self, other):
    """ True als twee deeltjes overlappen (afstand < som van de stralen) """
    dx = self.r[0] - other.r[0]
    dy = self.r[1] - other.r[1]
    rr = self.R + other.R
    return dx**2 + dy**2 < rr**2

def particle_collision(p1: ParticleClass, p2: ParticleClass):
    """ past snelheden aan uitgaande van overlap """
    m1, m2 = p1.m, p2.m
    delta_r = p1.r - p2.r
    delta_v = p1.v - p2.v
    dot_product = np.dot(delta_r, delta_v)

    # Als deeltjes van elkaar weg bewegen dan geen botsing
    # ('='-teken voorkomt ook problemen als delta_r == \vec{0})
    if dot_product >= 0:
        return False

    distance_squared = np.dot(delta_r, delta_r)
    # Botsing oplossen volgens elastische botsing in 2D
    p1.v -= 2 * m2 / (m1 + m2) * dot_product / distance_squared * delta_r
    p2.v += 2 * m1 / (m1 + m2) * dot_product / distance_squared * delta_r
    return True

def handle_collisions(particles):
    """ alle onderlinge botsingen afhandelen; geeft aantal botsingen terug """
    num_collisions = 0
    num_particles = len(particles)
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if collide_detection(particles[i], particles[j]):
                if particle_collision(particles[i], particles[j]):
                    num_collisions += 1
    return num_collisions


# tracken van het zware deeltje
track_x = []
track_y = []

# beginsituatie: N-1 lichte deeltjes en 1 zwaar deeltje in het midden
def maak_deeltjes():
    deeltjes = []
    for _ in range(N-1):
        hoek = np.random.uniform(0, 2*np.pi)
        startpos = (Box_length_0 - 0.5) * np.random.uniform(-1, 1, 2)
        snelheid = v_0 * np.array([np.cos(hoek), np.sin(hoek)])
        deeltjes.append(ParticleClass(m=1.0, v=snelheid, r=startpos, R=.5, c='blue'))
    deeltjes.append(ParticleClass(m=20.0, v=[0, 0], r=[0, 0], R=.5, c='red'))
    return deeltjes

particles = maak_deeltjes()

# tijdstippen
t = []
# totale kinetische energie per stap
energie = []
# aantal botsingen per stap
n_botsingen = []

for i in range(400):
    t.append(i * dt)
    
    for p in particles:
        # Update positie
        p.update_position()
        # Wandbotsing werkt per deeltje
        p.boxcollision()
        
    n_botsingen.append(handle_collisions(particles))

    energie.append(sum(p.kin_energy for p in particles))
    
    track_x.append(particles[N-1].r[0])
    track_y.append(particles[N-1].r[1])

# afstand tussen begin- en eindpunt van het zware deeltje
dx = track_x[-1] - track_x[0]
dy = track_y[-1] - track_y[0]
verplaatsing = np.sqrt(dx**2 + dy**2)

print(f"afstand start-eind zwaar deeltje: {verplaatsing:.2f}")
print(f"totaal aantal botsingen: {np.sum(n_botsingen)}")
print(f"energie begin: {energie[0]:.3f}, eind: {energie[-1]:.3f}")

plt.figure()
plt.plot(track_x,track_y,'r')
plt.scatter([track_x[0], track_x[-1]], [track_y[0], track_y[-1]], c=['g', 'k'], zorder=3)
plt.annotate('start', (track_x[0], track_y[0]), color='g')
plt.annotate('eind', (track_x[-1], track_y[-1]), color='k')
plt.axis([-Box_length_0, Box_length_0, -Box_length_0, Box_length_0])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-input", "remove-output"]
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML, display

# ---- beginsituatie bewaren, zodat elke run hetzelfde start ----
particles_start = copy.deepcopy(particles)

# ---- tracking arrays ----
track_x, track_y = [], []          # zwaar deeltje (particles[N-1])
track_x2, track_y2 = [], []        # tweede deeltje (particles[0])
botsingen = []                     # aantal botsingen per tijdstap

# ---- figure setup ----
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-Box_size_0/2, Box_size_0/2)
ax.set_ylim(-Box_size_0/2, Box_size_0/2)
ax.set_aspect('equal')

dots, = ax.plot([], [], 'ro', ms=18)        # alle deeltjes
heavy_dot, = ax.plot([], [], 'ko', ms=20)   # zwaar deeltje
light_dot, = ax.plot([], [], 'bo', ms=18)   # tweede deeltje
heavy_path, = ax.plot([], [], 'k-', lw=1, alpha=0.8)
light_path, = ax.plot([], [], 'b-', lw=1, alpha=0.8)

def init():
    global particles
    particles = copy.deepcopy(particles_start)   # terug naar beginsituatie
    for lijst in (track_x, track_y, track_x2, track_y2, botsingen):
        lijst.clear()
    dots.set_data([], [])
    heavy_dot.set_data([], [])
    light_dot.set_data([], [])
    heavy_path.set_data([], [])
    light_path.set_data([], [])
    return dots, heavy_dot, light_dot, heavy_path, light_path

def update(frame):
    for p in particles:
        p.update_position()
        p.boxcollision()

    n = handle_collisions(particles)
    botsingen.append(n if n is not None else 0)

    track_x.append(particles[N-1].r[0])
    track_y.append(particles[N-1].r[1])
    track_x2.append(particles[0].r[0])
    track_y2.append(particles[0].r[1])

    dots.set_data([p.r[0] for p in particles], [p.r[1] for p in particles])
    heavy_dot.set_data([track_x[-1]], [track_y[-1]])
    light_dot.set_data([track_x2[-1]], [track_y2[-1]])
    heavy_path.set_data(track_x, track_y)
    light_path.set_data(track_x2, track_y2)

    return dots, heavy_dot, light_dot, heavy_path, light_path

ani = FuncAnimation(fig, update, frames=400, init_func=init, blit=True, interval=50)

display(HTML(ani.to_jshtml()))      # animatie in het notebook
ani.save("deeltjes.gif", writer="pillow", fps=20)   # zelfde run, opnieuw vanaf start
plt.close(fig)

print("totaal aantal botsingen:", sum(botsingen))

# HD-video (vereist ffmpeg):
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Jij'), bitrate=5000)
# ani.save("deeltjes_hd.mp4", writer=writer, dpi=200)
```

```{exercise} Brownian motion in beeld
:label: ex-brownian-3
- Draai de simulatie een keer en bestudeer de output.
- Voeg zelf een tweede tracking toe van een licht deeltje en verbeter de plot.
- Wat zijn overeenkomsten en verschillen tussen de beweging van de twee deeltjes?
- Wat valt je op als je de simulatie een aantal keer runt?
```

```{solution} ex-brownian-3
**Overeenkomsten:** Beide deeltjes bewegen in rechte stukjes die bij elke botsing van richting veranderen.

**Verschillen:** Het lichte deeltje beweegt snel door de hele box. Het zware deeltje ($m = 20$) maakt kleine stapjes rond het midden, omdat elke botsing zijn snelheid maar een klein beetje verandert ($v_\text{zwaar} \approx v_\text{licht}/\sqrt{20} \approx 0{,}2$).

**Meerdere keren runnen:** Elke run geeft een ander pad, maar het lichte deeltje komt steeds veel verder dan het zware.
 
Elke keer dat er een botsing plaatsvindt, zou de counter met 1 omhoog moeten gaan.
Het beste is als het aantal botsingen opgeslagen zoud worden in een array zodat je het aantal botsingen als functie van de tijd kunt weergeven.

```{exercise}
Pas bovenstaand idee toe in de eerder gemaakte code.
Plot hieronder het aantal botsingen als functie van de tijd.
```

```python
plt.figure()
plt.plot(track_x2, track_y2, color='tab:blue', lw=1, alpha=0.6, label='licht deeltje')
plt.plot(track_x, track_y, color='hotpink', lw=2, label='zwaar deeltje')
plt.plot(track_x[0], track_y[0], 'o', color='green', label='start zwaar deeltje')
plt.xlim(-Box_length_0, Box_length_0)
plt.ylim(-Box_length_0, Box_length_0)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()
```

```{warning} 🌶 Let op!
:icon: false
De onderstaande opdrachten vallen buiten de stof maar tellen mee als je excellent wilt behalen.
```

In zulke fysica modellen is de afgelegde weg (afstand tussen begin en eindpunt) van belang.
Deze afgelegde weg zegt iets over de snelheid van difussie.
Idealiter bekijken we een histogram.
Maar voor een histogram hebben we veel deeltjes nodig.



```{exercise} Afgelegde weg 🌶
:label: ex-brownian-4

- Maak een simulatie met 361 deeltjes, waarvan 1 zwaar deeltje.
- Houd rekening met de boxgrootte, deze moet mee schalen!
- Maak een histogram van de afgelegde weg voor alle deeltjes. 
- Geef de afgelegde weg van het grote deeltje duidelijk aan.
```


```python tags=["NB2_brownian_distance_histogram"] editable=true slideshow={"slide_type": ""}
# Parameters
N = 361
v_0 = 1
dt = 0.04
R = 0.5
n_steps = 400

# Doos schaalt mee: zelfde oppervlak per deeltje als bij N = 40 in een 10x10 doos
doos_grootte = 10 * np.sqrt(N / 40)
doos_lengte = doos_grootte / 2
# boxcollision() in ParticleClass gebruikt de naam Box_length
Box_length = doos_lengte


def maak_deeltjes():
    """ N-1 lichte deeltjes op een 19x19 rooster, zwaar deeltje in het midden """
    n_zijde = 19
    afstand_rooster = doos_grootte / n_zijde
    deeltjes = []
    for i in range(n_zijde):
        for j in range(n_zijde):
            # Middelste plek overslaan, dat is voor het zware deeltje
            if i == 9 and j == 9:
                continue
            x = (i - 9) * afstand_rooster
            y = (j - 9) * afstand_rooster
            hoek = np.random.uniform(0, 2*np.pi)
            v = [v_0 * np.cos(hoek), v_0 * np.sin(hoek)]
            deeltjes.append(ParticleClass(m=1.0, v=v, r=[x, y], R=R, c='blue'))
    
    # Zwaar deeltje als laatste in de lijst
    deeltjes.append(ParticleClass(m=20.0, v=[0, 0], r=[0, 0], R=R, c='red'))
    return deeltjes


def simuleer(deeltjes, n_steps):
    """ laat de deeltjes n_steps tijdstappen bewegen en botsen """
    for stap in range(n_steps):
        for p in deeltjes:
            p.update_position()
            p.boxcollision()
        handle_collisions(deeltjes)


def afgelegde_weg(deeltjes, start):
    """ afstand tussen begin- en eindpunt voor elk deeltje """
    afstand = []
    for p, r0 in zip(deeltjes, start):
        dx = p.r[0] - r0[0]
        dy = p.r[1] - r0[1]
        afstand.append(np.sqrt(dx**2 + dy**2))
    
    return np.array(afstand)


def plot_histogram(afstand_licht, afstand_zwaar):
    """ histogram van de lichte deeltjes, zwaar deeltje als lijn """
    plt.figure()
    plt.hist(afstand_licht, bins=30, color='tab:blue', edgecolor='k', label='lichte deeltjes')
    plt.axvline(afstand_zwaar, color='hotpink', lw=3, label=f'zwaar deeltje = {afstand_zwaar:.2f}')
    plt.xlabel('afgelegde weg')
    plt.ylabel('aantal deeltjes')
    plt.legend()
    plt.show()


# Uitvoeren
particles = maak_deeltjes()
start = [p.r.copy() for p in particles]
simuleer(particles, n_steps)

afstand = afgelegde_weg(particles, start)

plot_histogram(afstand[:-1], afstand[-1])
print(f"gemiddelde afgelegde weg licht: {np.mean(afstand[:-1]):.2f}")
print(f"afgelegde weg zwaar deeltje: {afstand[-1]:.2f}")
```

En nu we toch bezig zijn met twee verschillende deeltjes.... 

We kunnen twee "groepen" van deeltjes aanmaken, elk  met een andere massa. Als we dan de zwaartekracht aan zetten, dan zouden we verwachten dat de lichtere deeltjes boven komen "drijven".

```{exercise} Onderzoek dit vermoeden 🌶
- maak daartoe de box 2x zo hoog als breed
- verdubbel het totaal aantal deeltjes
- zet een artificieel grote zwaartekracht aan
```

```python
# Twee groepen deeltjes met zwaartekracht

# Parameters
doos_breedte = 10
# Doos 2x zo hoog als breed
doos_hoogte = 2 * doos_breedte
# Twee keer zoveel deeltjes: 40 licht + 40 zwaar
N = 80
m_licht = 1.0
m_zwaar = 5.0
v_0 = 1
R = 0.5
dt = 0.04
# Insane grote zwaartekracht
g = 0.05
n_steps = 4000


class ParticleGravClass:
    def __init__(self, m, v, r, R, c):
        self.m = m
        self.v = np.array(v, dtype=float)
        self.r = np.array(r, dtype=float)
        self.R = R
        self.c = c

    def update_position(self):
        # Zwaartekracht trekt de snelheid elke stap omlaag
        self.v[1] -= g * dt
        self.r += self.v * dt

    def boxcollision(self):
        # Zijwanden
        if abs(self.r[0]) > doos_breedte/2 - self.R and self.r[0] * self.v[0] > 0:
            self.v[0] = -self.v[0]
        
        # Bodem
        if self.r[1] < self.R and self.v[1] < 0:
            self.v[1] = -self.v[1]
        
        # Plafond
        if self.r[1] > doos_hoogte - self.R and self.v[1] > 0:
            self.v[1] = -self.v[1]


def maak_deeltjes():
    """ 8x10 rooster, licht en zwaar om en om, allemaal dezelfde kinetische energie """
    deeltjes = []
    for i in range(8):
        for j in range(10):
            x = (i - 3.5) * doos_breedte / 8
            y = (j + 0.5) * doos_hoogte / 10
            # Dambordpatroon: zwaar als i + j oneven is
            if (i + j) % 2 == 1:
                m, kleur = m_zwaar, 'red'
            else:
                m, kleur = m_licht, 'blue'
            
            # Zware deeltjes trager, zodat 1/2 m v^2 gelijk is
            v = v_0 * np.sqrt(m_licht / m)
            hoek = np.random.uniform(0, 2*np.pi)
            deeltjes.append(ParticleGravClass(m, [v*np.cos(hoek), v*np.sin(hoek)], [x, y], R, kleur))
    
    return deeltjes


def gemiddelde_hoogte(deeltjes, massa):
    """ gemiddelde y van alle deeltjes met een bepaalde massa """
    hoogtes = [p.r[1] for p in deeltjes if p.m == massa]
    
    return np.mean(hoogtes)

def simuleer(deeltjes, n_steps):
    """ laat de deeltjes bewegen en houdt de gemiddelde hoogte per groep bij """
    hoogte_licht = []
    hoogte_zwaar = []
    for stap in range(n_steps):
        for p in deeltjes:
            p.update_position()
            p.boxcollision()
        handle_collisions(deeltjes)
        hoogte_licht.append(gemiddelde_hoogte(deeltjes, m_licht))
        hoogte_zwaar.append(gemiddelde_hoogte(deeltjes, m_zwaar))
    
    return hoogte_licht, hoogte_zwaar


# Simulatie
particles = maak_deeltjes()
hoogte_licht, hoogte_zwaar = simuleer(particles, n_steps)

print(f"gemiddelde hoogte licht: {np.mean(hoogte_licht[n_steps//2:]):.1f}")
print(f"gemiddelde hoogte zwaar: {np.mean(hoogte_zwaar[n_steps//2:]):.1f}")

# Plots
tijd = np.arange(n_steps) * dt
plt.figure(figsize=(10, 5))

# Gemiddelde hoogte per groep tegen de tijd
plt.subplot(1, 2, 1)
plt.plot(tijd, hoogte_licht, color='tab:blue', label='licht')
plt.plot(tijd, hoogte_zwaar, color='hotpink', label='zwaar')
plt.xlabel('tijd')
plt.ylabel('gemiddelde hoogte')
plt.legend()

# Eindsituatie in de doos
plt.subplot(1, 2, 2)
for p in particles:
    kleur = 'tab:blue' if p.m == m_licht else 'hotpink'
    plt.gca().add_patch(plt.Circle(p.r, p.R, color=kleur))
plt.xlim(-doos_breedte/2, doos_breedte/2)
plt.ylim(0, doos_hoogte)
plt.show()
```

```python

```

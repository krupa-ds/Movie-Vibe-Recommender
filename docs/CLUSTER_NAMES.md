# Film Cluster Taxonomy

A curated semantic taxonomy of 60 film clusters derived from content-based analysis of 4,305 films across genres, themes, and user-generated tags.

---

## Overview

**Clustering Method**: MiniBatchKMeans (k=60) on L2-normalized content features  
**Features**: 202 dimensions (27 genres + 15 themes + 160 filtered tags)  
**Silhouette Score**: 0.115 (low but semantically meaningful)  
**Average Cluster Size**: 72 films  

**Note**: Cluster names were created using LLM's and may not perfectly align with algorithmic groupings. The content-based clustering identifies films with similar genre/theme/tag patterns, which sometimes produces unexpected but semantically coherent groupings.

---

## Key Observations

**Largest Clusters**:
1. Cluster 13 (238 films) - Broad dramatic range
2. Cluster 32 (171 films) - Powerful diverse dramas
3. Cluster 4 (157 films) - Profound human dramas

**Smallest Clusters**:
1. Cluster 37 (18 films) - Intimate character studies
2. Cluster 51 (28 films) - War and romantic dramas
3. Cluster 57 (30 films) - Intense surreal dramas

**Dominant Themes Across All Clusters**:
- "Moving relationship stories" appears in 58/60 clusters
- "Humanity and the world around us" appears in 55/60 clusters
- Drama is the dominant genre in all clusters

**Note**: The prevalence of "Drama" and "Moving relationship stories" suggests the dataset is heavily weighted toward serious, emotionally-driven cinema rather than pure action, horror, or comedy films.

---

## Usage in Recommendation System

- **Diversity Control**: Maximum 3 films per cluster in recommendations
- **Taste Mapping**: 2D t-SNE visualization of cluster space
- **Similar Films**: Backend finds films from same/adjacent clusters
- **Search-to-Rate**: Initial 5 films selected from top clusters near searched film

---

## Complete Cluster Taxonomy

### **Cluster 0: Human Passion and Artistic Struggle**
Epic dramas exploring human relationships and philosophical depth.

**Representative Films**:
- Ran (1985)
- Satantango (1994)
- City Lights (1931)
- The Weeping Meadow (2004)
- Das Boot (1981)

**Size**: 76 films  
**Dominant Genres**: Drama (65%), Comedy (25%), Thriller (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world around us, Heartbreak and suffering

---

### **Cluster 1: Undead Chaos and Dark Comedy**
Emotionally complex dramas with moments of levity and darkness.

**Representative Films**:
- Life Is Beautiful (1997)
- All That Jazz (1979)
- Monster (2023)
- The Celebration (1998)
- Brief Encounter (1945)

**Size**: 53 films  
**Dominant Genres**: Drama (70%), Comedy (30%), Thriller (25%)  
**Key Themes**: Moving relationship stories, Humanity and the world around us, Crude humor and satire

---

### **Cluster 2: Intimate Emotional Dramas of the Heart**
Character-driven dramas with strong emotional cores and family dynamics.

**Representative Films**:
- The Godfather Part II (1974)
- Cinema Paradiso (1988)
- The Passion of Joan of Arc (1928)
- Farewell My Concubine (1993)
- Singin' in the Rain (1952)

**Size**: 99 films  
**Dominant Genres**: Drama (79%), Crime (20%), Comedy (17%)  
**Key Themes**: Moving relationship stories, Humanity and the world around us, Family stories

---

### **Cluster 3: Adrenaline-Fueled Action Spectacles**
Thought-provoking dramas with philosophical and existential themes.

**Representative Films**:
- Mishima: A Life in Four Chapters (1985)
- My Night at Maud's (1969)
- The Worst Person in the World (2021)
- Kneecap (2024)
- Encounters at the End of the World (2007)

**Size**: 35 films  
**Dominant Genres**: Drama (69%), Comedy (31%), Action (17%)  
**Key Themes**: Moving relationship stories, Surreal visions, Transgression

---

### **Cluster 4: Charming Hearts and Comic Tales**
Profound human dramas exploring life's complexities and moral questions.

**Representative Films**:
- Yi Yi (2000)
- Shoah (1985)
- Hoop Dreams (1994)
- Princess Mononoke (1997)
- The Seventh Seal (1957)

**Size**: 157 films  
**Dominant Genres**: Drama (66%), Comedy (28%), Thriller (19%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Family drama

---

### **Cluster 5: Criminal Underworld Dramas**
Intimate character studies of ordinary people and quiet emotional depth.

**Representative Films**:
- Nights of Cabiria (1957)
- HOMECOMING: A film by Beyoncé (2019)
- Still Walking (2008)
- The Lives of Others (2006)
- Umberto D. (1952)

**Size**: 55 films  
**Dominant Genres**: Drama (73%), Comedy (22%), Romance (20%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Family stories

---

### **Cluster 6: Dark Transgressive Thrillers**
Artistic and visually distinctive dramas with romantic undertones.

**Representative Films**:
- Persona (1966)
- In the Mood for Love (2000)
- Where Is the Friend's House? (1987)
- Dreams (1990)
- On the Waterfront (1954)

**Size**: 78 films  
**Dominant Genres**: Drama (54%), Comedy (24%), Romance (23%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Psychological thriller

---

### **Cluster 7: Outlaws, Dust, and Redemption**
Tender family dramas and relationship-focused narratives.

**Representative Films**:
- Autumn Sonata (1978)
- The Gleaners and I (2000)
- Make Way for Tomorrow (1937)
- Unforgiven (1992)
- North by Northwest (1959)

**Size**: 98 films  
**Dominant Genres**: Drama (66%), Comedy (24%), Romance (24%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Family stories

---

### **Cluster 8: Humanity's Reckoning with War**
Fantasy and romantic dramas with emotional depth.

**Representative Films**:
- Spirited Away (2001)
- Fanny and Alexander (1982)
- Ali: Fear Eats the Soul (1974)
- Anatomy of a Fall (2023)
- Orpheus (1950)

**Size**: 43 films  
**Dominant Genres**: Drama (70%), Romance (33%), Comedy (26%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Romance

---

### **Cluster 9: Profound Human Journeys**
Morally complex dramas exploring justice and human nature.

**Representative Films**:
- High and Low (1963)
- Schindler's List (1993)
- The Apartment (1960)
- Inglourious Basterds (2009)
- Life, and Nothing More… (1992)

**Size**: 71 films  
**Dominant Genres**: Drama (70%), Comedy (25%), Romance (25%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 10: Comedic Brilliance and Satirical Wit**
Intense dramatic works exploring violence, humanity, and moral complexity.

**Representative Films**:
- The Human Condition II: Road to Eternity (1959)
- O.J.: Made in America (2016)
- Paths of Glory (1957)
- Oldboy (2003)
- The 400 Blows (1959)

**Size**: 142 films  
**Dominant Genres**: Drama (71%), Comedy (25%), Thriller (18%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Violence and transgression

---

### **Cluster 11: Transgressive Class Conflict Thrillers**
War films and coming-of-age stories with romantic elements.

**Representative Films**:
- Apocalypse Now (1979)
- Nobody Knows (2004)
- Before Sunrise (1995)
- The Deer Hunter (1978)
- Stand by Me (1986)

**Size**: 50 films  
**Dominant Genres**: Drama (60%), Comedy (32%), Romance (24%)  
**Key Themes**: Moving relationship stories, Violence and transgression, Crude humor

---

### **Cluster 12: The Art of Making Music**
Dark, psychologically complex dramas with crime elements.

**Representative Films**:
- Perfect Blue (1997)
- Three Colours: Red (1994)
- Underground (1995)
- Throne of Blood (1957)
- The Turin Horse (2011)

**Size**: 62 films  
**Dominant Genres**: Drama (74%), Crime (29%), Thriller (23%)  
**Key Themes**: Moving relationship stories, Heartbreak and suffering, Humanity and the world

---

### **Cluster 13: Profound Family Heartbreak and Humanity**
The largest cluster: diverse dramas with broad thematic range.

**Representative Films**:
- The Ascent (1977)
- Red Beard (1965)
- All About Eve (1950)
- Dead Poets Society (1989)
- The Silence of the Lambs (1991)

**Size**: 238 films (largest cluster)  
**Dominant Genres**: Drama (62%), Comedy (27%), Thriller (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Crude humor and satire

---

### **Cluster 14: Darkly Gripping Crime Mysteries**
Emotionally charged dramas with horror and supernatural elements.

**Representative Films**:
- A Woman Under the Influence (1974)
- Witness for the Prosecution (1957)
- La Commune (Paris, 1871) (2000)
- Kwaidan (1964)
- Eternal Sunshine of the Spotless Mind (2004)

**Size**: 48 films  
**Dominant Genres**: Drama (69%), Romance (25%), Horror (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Horror

---

### **Cluster 15: Whimsical Family Adventures**
Serious dramas exploring human suffering and social issues.

**Representative Films**:
- The Human Condition III: A Soldier's Prayer (1961)
- Rififi (1955)
- David Byrne's American Utopia (2020)
- Poetry (2010)
- 12 Years a Slave (2013)

**Size**: 65 films  
**Dominant Genres**: Drama (72%), Comedy (28%), Romance (17%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Family stories

---

### **Cluster 16: War's Human Tragedy**
Artistic dramas with thriller and adventure elements.

**Representative Films**:
- Eternity and a Day (1998)
- Amadeus (1984)
- The Prestige (2006)
- Sorcerer (1977)
- Moonlight (2016)

**Size**: 47 films  
**Dominant Genres**: Drama (70%), Thriller (21%), Adventure (21%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Surreal visions

---

### **Cluster 17: Emotional Stories of Human Struggle**
Mixed dramas with comedic and dark elements.

**Representative Films**:
- The Human Condition I: No Greater Love (1959)
- Spider-Man: Across the Spider-Verse (2023)
- Twin Peaks: Fire Walk with Me (1992)
- The Iron Giant (1999)
- Napoleon (1927)

**Size**: 78 films  
**Dominant Genres**: Drama (54%), Comedy (24%), Romance (19%)  
**Key Themes**: Crude humor and satire, Moving relationship stories, Violence and transgression

---

### **Cluster 18: Spiritual Reckoning and Human Mortality**
Character-driven dramas about passion and human connection.

**Representative Films**:
- Whiplash (2014)
- Do the Right Thing (1989)
- Dersu Uzala (1975)
- Chungking Express (1994)
- The Life and Death of Colonel Blimp (1943)

**Size**: 63 films  
**Dominant Genres**: Drama (67%), Romance (25%), Comedy (25%)  
**Key Themes**: Moving relationship stories, Heartbreak and suffering, Humanity and the world

---

### **Cluster 19: Dark Psychological Horror Visions**
Varied dramas with thriller and adventure elements.

**Representative Films**:
- Spider-Man: Into the Spider-Verse (2018)
- Mirror (1975)
- Tokyo Twilight (1957)
- For a Few Dollars More (1965)
- Sunrise: A Song of Two Humans (1927)

**Size**: 53 films  
**Dominant Genres**: Drama (62%), Thriller (30%), Adventure (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Family drama

---

### **Cluster 20: Animated Enchantment and Wonder**
Moral and philosophical dramas exploring justice and violence.

**Representative Films**:
- Stalker (1979)
- Judgment at Nuremberg (1961)
- A Special Day (1977)
- Dog Day Afternoon (1975)
- The Man Who Shot Liberty Valance (1962)

**Size**: 109 films  
**Dominant Genres**: Drama (67%), Comedy (27%), Thriller (23%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Violence and transgression

---

### **Cluster 21: Gritty Crime and Moral Conflict**
Intimate dramas with crime elements and moral complexity.

**Representative Films**:
- Le Trou (1960)
- The Beaches of Agnès (2008)
- Perfect Days (2023)
- Shoplifters (2018)
- Kagemusha (1980)

**Size**: 90 films  
**Dominant Genres**: Drama (64%), Comedy (29%), Crime (24%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Violence and transgression

---

### **Cluster 22: Quirky Coming-of-Age Comedies**
Comedic dramas with romantic elements.

**Representative Films**:
- Paris Is Burning (1990)
- Dune: Part Two (2024)
- Ordet (1955)
- No Country for Old Men (2007)
- The Shop on Main Street (1965)

**Size**: 61 films  
**Dominant Genres**: Drama (69%), Comedy (39%), Romance (28%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Relationship comedy

---

### **Cluster 23: Surreal Minds Bending Reality**
Comedic dramas with absurdist and satirical elements.

**Representative Films**:
- Landscape in the Mist (1988)
- Swing Girls (2004)
- Raise the Red Lantern (1991)
- Tampopo (1985)
- Children of Paradise (1945)

**Size**: 40 films  
**Dominant Genres**: Drama (58%), Comedy (25%), Thriller (25%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Gags

---

### **Cluster 24: Intimate Dramas of Human Suffering**
Varied dramas exploring humanity and relationships.

**Representative Films**:
- The Young Girls of Rochefort (1967)
- Werckmeister Harmonies (2000)
- Ugetsu (1953)
- Saving Private Ryan (1998)
- The Sting (1973)

**Size**: 70 films  
**Dominant Genres**: Drama (66%), Comedy (29%), Thriller (23%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Crude humor and satire

---

### **Cluster 25: Mysteries of the Human Psyche**
Thrillers and action films with dramatic elements.

**Representative Films**:
- Night and Fog (1956)
- Z (1969)
- Amélie (2001)
- The Matrix (1999)
- Raiders of the Lost Ark (1981)

**Size**: 42 films  
**Dominant Genres**: Drama (36%), Thriller (33%), Comedy (33%)  
**Key Themes**: Moving relationship stories, High speed and special ops, Humanity and the world

---

### **Cluster 26: Surreal Dreams of Human Experience**
Philosophical dramas about mortality and human existence.

**Representative Films**:
- As I Was Moving Ahead, Occasionally I Saw Brief Glimpses of Beauty (2000)
- Sing Sing (2023)
- The Father (2020)
- Double Indemnity (1944)
- Magnolia (1999)

**Size**: 102 films  
**Dominant Genres**: Drama (65%), Comedy (23%), Romance (22%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Heartbreak and suffering

---

### **Cluster 27: Political Power and Corruption**
Emotionally powerful dramas with satirical elements.

**Representative Films**:
- It's Such a Beautiful Day (2012)
- Dancer in the Dark (2000)
- Au Revoir les Enfants (1987)
- Day for Night (1973)
- Faces Places (2017)

**Size**: 80 films  
**Dominant Genres**: Drama (55%), Comedy (33%), Thriller (16%)  
**Key Themes**: Moving relationship stories, Heartbreak and suffering, Crude humor and satire

---

### **Cluster 28: Intimate Truths of Human Struggle**
Dramatic comedies exploring human nature and society.

**Representative Films**:
- One Flew Over the Cuckoo's Nest (1975)
- Se7en (1995)
- To Be or Not to Be (1942)
- Malcolm X (1992)
- Devils on the Doorstep (2000)

**Size**: 61 films  
**Dominant Genres**: Drama (62%), Comedy (31%), Romance (25%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Humanity and the world

---

### **Cluster 29: Haunted Shadows and Spectral Dread**
Crime dramas and romantic comedies.

**Representative Films**:
- The Godfather (1972)
- Before Sunset (2004)
- The Hunt (2012)
- The Devils (1971)
- Comrades, Almost a Love Story (1996)

**Size**: 73 films  
**Dominant Genres**: Drama (62%), Comedy (34%), Romance (26%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Humanity and the world

---

### **Cluster 30: Musical Joy and Romance**
Tense dramas and thrillers with violent elements.

**Representative Films**:
- Once Upon a Time in the West (1968)
- The Wages of Fear (1953)
- Minding the Gap (2018)
- Solaris (1972)
- F for Fake (1973)

**Size**: 78 films  
**Dominant Genres**: Drama (63%), Comedy (26%), Thriller (24%)  
**Key Themes**: Moving relationship stories, Violence and transgression, Heartbreak and suffering

---

### **Cluster 31: High-Stakes Tension and Intrigue**
Diverse dramas exploring humanity and suffering.

**Representative Films**:
- Psycho (1960)
- The King of Comedy (1982)
- A Moment of Innocence (1996)
- The Grapes of Wrath (1940)
- My Neighbor Totoro (1988)

**Size**: 41 films  
**Dominant Genres**: Drama (63%), Comedy (34%), Romance (17%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Heartbreak and suffering

---

### **Cluster 32: Psychological Horror Masterpieces**
The second-largest cluster: diverse powerful dramas.

**Representative Films**:
- The Shawshank Redemption (1994)
- The Battle of Algiers (1966)
- Incendies (2010)
- The Lord of the Rings: The Fellowship of the Ring (2001)
- Senna (2010)

**Size**: 171 films (second-largest)  
**Dominant Genres**: Drama (56%), Comedy (23%), Thriller (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 33: Epic Heroic Adventures**
Science fiction and fantasy adventures with comedy.

**Representative Films**:
- Paper Moon (1973)
- Sherlock Jr. (1924)
- Star Wars (1977)
- Vampire Hunter D: Bloodlust (2000)
- Brazil (1985)

**Size**: 40 films  
**Dominant Genres**: Drama (43%), Comedy (35%), Science Fiction (25%)  
**Key Themes**: Moving relationship stories, Crude humor and satire, Gags

---

### **Cluster 34: Cosmic Horror and Alien Encounters**
Philosophical dramas with surreal elements.

**Representative Films**:
- The Good, the Bad and the Ugly (1966)
- It's a Wonderful Life (1946)
- Portrait of a Lady on Fire (2019)
- Love Exposure (2008)
- For Sama (2019)

**Size**: 41 films  
**Dominant Genres**: Drama (61%), Comedy (29%), Romance (29%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Surreal visions

---

### **Cluster 35: Comedic Action and Physical Humor**
Crime dramas with emotional depth.

**Representative Films**:
- Dear Zachary: A Letter to a Son About His Father (2008)
- An Elephant Sitting Still (2018)
- Good Will Hunting (1997)
- World of Tomorrow (2015)
- Samsara (2011)

**Size**: 64 films  
**Dominant Genres**: Drama (64%), Crime (27%), Comedy (27%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Violence and transgression

---

### **Cluster 36: Triumph Against All Odds**
Socially conscious thrillers and crime dramas.

**Representative Films**:
- Parasite (2019)
- Howl's Moving Castle (2004)
- Fireworks (1997)
- The Thin Blue Line (1988)
- House of Hummingbird (2018)

**Size**: 36 films  
**Dominant Genres**: Drama (69%), Thriller (22%), Crime (22%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Crime

---

### **Cluster 37: Documenting Justice and Resistance**
Intimate character studies and relationship dramas.

**Representative Films**:
- Joint Security Area (2000)
- Ivan's Childhood (1962)
- What Ever Happened to Baby Jane? (1962)
- Ghost Dog: The Way of the Samurai (1999)
- Babette's Feast (1987)

**Size**: 18 films (smallest cluster)  
**Dominant Genres**: Drama (78%), Romance (22%), Action (22%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Crude humor and satire

---

### **Cluster 38: War's Moral Reckoning**
Morally complex dramas exploring violence and family.

**Representative Films**:
- There Will Be Blood (2007)
- The Cranes Are Flying (1957)
- Late Spring (1949)
- La Notte (1961)
- The Cremator (1969)

**Size**: 54 films  
**Dominant Genres**: Drama (69%), Comedy (30%), Romance (28%)  
**Key Themes**: Humanity and the world, Violence and transgression, Family drama

---

### **Cluster 39: Intimate Stories of Desire**
Epic and emotionally powerful dramas.

**Representative Films**:
- The Empire Strikes Back (1980)
- The White Ribbon (2009)
- The Long Goodbye (1973)
- Turtles Can Fly (2004)
- The Leopard (1963)

**Size**: 92 films  
**Dominant Genres**: Drama (64%), Comedy (23%), Romance (16%)  
**Key Themes**: Moving relationship stories, Heartbreak and suffering, Family drama

---

### **Cluster 40: Transgressive Human Drama**
Intense dramas exploring violence and humanity.

**Representative Films**:
- Come and See (1985)
- Sansho the Bailiff (1954)
- Paris, Texas (1984)
- Fantastic Mr. Fox (2009)
- Casablanca (1942)

**Size**: 94 films  
**Dominant Genres**: Drama (60%), Comedy (23%), Romance (20%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Violence and transgression

---

### **Cluster 41: Coming-of-Age Human Dramas**
Character-driven relationship dramas.

**Representative Films**:
- Scenes from a Marriage (1974)
- Chinatown (1974)
- The Treasure of the Sierra Madre (1948)
- Whisper of the Heart (1995)
- Synecdoche, New York (2008)

**Size**: 89 films  
**Dominant Genres**: Drama (72%), Comedy (25%), Romance (19%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 42: Epic Science Fiction Adventures**
Philosophical and emotionally intense dramas.

**Representative Films**:
- Memories of Murder (2003)
- 2001: A Space Odyssey (1968)
- Amour (2012)
- The Rocky Horror Picture Show (1975)
- Fruitvale Station (2013)

**Size**: 47 films  
**Dominant Genres**: Drama (72%), Comedy (21%), Romance (19%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 43: Urban Crime and Social Decay**
Dramas exploring humanity and social issues.

**Representative Films**:
- Harakiri (1962)
- La Haine (1995)
- I Am Cuba (1964)
- Interstellar (2014)
- The Red Shoes (1948)

**Size**: 81 films  
**Dominant Genres**: Drama (69%), Comedy (24%), Romance (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Crude humor and satire

---

### **Cluster 44: Guns, Grit, and Glory**
Family-centered dramas with emotional depth.

**Representative Films**:
- Grave of the Fireflies (1988)
- Apur Sansar (1959)
- Akira (1988)
- Back to the Future (1985)
- Good Morning (1959)

**Size**: 60 films  
**Dominant Genres**: Drama (68%), Comedy (27%), Romance (20%)  
**Key Themes**: Moving relationship stories, Family drama, Humanity and the world

---

### **Cluster 45: Humanity's Poetic Documentary Journey**
Artistic and surreal dramatic works.

**Representative Films**:
- The Pianist (2002)
- Time of the Gypsies (1988)
- The Cook, the Thief, His Wife & Her Lover (1989)
- Spring, Summer, Fall, Winter... and Spring (2003)
- Adaptation. (2002)

**Size**: 45 films  
**Dominant Genres**: Drama (67%), Crime (20%), Thriller (18%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Surreal visions

---

### **Cluster 46: Epic Fantasy Adventures**
Action-driven crime dramas.

**Representative Films**:
- City of God (2002)
- Cure (1997)
- Human (2015)
- Ulysses' Gaze (1995)
- Mustang (2015)

**Size**: 39 films  
**Dominant Genres**: Drama (51%), Action (26%), Crime (23%)  
**Key Themes**: Epic heroes, Moving relationship stories, Humanity and the world

---

### **Cluster 47: Humanity's Cosmic Reckoning**
Epic human dramas with historical and philosophical depth.

**Representative Films**:
- Seven Samurai (1954)
- The Great Dictator (1940)
- To Live (1994)
- La Belle Noiseuse (1991)
- Marketa Lazarovà (1967)

**Size**: 46 films  
**Dominant Genres**: Drama (63%), Comedy (26%), Action (13%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Heartbreak and suffering

---

### **Cluster 48: Epic Historical Dramas**
Dark psychological thrillers with intense themes.

**Representative Films**:
- The Handmaiden (2016)
- Harlan County U.S.A. (1976)
- 4 Months, 3 Weeks and 2 Days (2007)
- Quo Vadis, Aida? (2020)
- Embrace of the Serpent (2015)

**Size**: 47 films  
**Dominant Genres**: Drama (66%), Thriller (28%), Comedy (28%)  
**Key Themes**: Heartbreak and suffering, Moving relationship stories, Psychological thriller

---

### **Cluster 49: Timeless Romance and Human Connection**
Diverse dramas exploring justice and human struggle.

**Representative Films**:
- 12 Angry Men (1957)
- A Brighter Summer Day (1991)
- Central Station (1998)
- Heat (1995)
- Ritual (2000)

**Size**: 79 films  
**Dominant Genres**: Drama (68%), Comedy (34%), Action (19%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Heartbreak and suffering

---

### **Cluster 50: Passionate Tales of the Heart**
Emotionally intense family dramas.

**Representative Films**:
- Neon Genesis Evangelion: The End of Evangelion (1997)
- The Look of Silence (2014)
- Eureka (2000)
- Evangelion: 3.0+1.0 Thrice Upon a Time (2021)
- All the Beauty and the Bloodshed (2022)

**Size**: 76 films  
**Dominant Genres**: Drama (70%), Thriller (25%), Comedy (24%)  
**Key Themes**: Moving relationship stories, Family stories, Heartbreak and suffering

---

### **Cluster 51: Heroic Comedy Adventures**
War and romantic dramas with emotional weight.

**Representative Films**:
- Full Metal Jacket (1987)
- Forrest Gump (1994)
- Pretty Village, Pretty Flame (1996)
- Kiki's Delivery Service (1989)
- Redline (2009)

**Size**: 28 films  
**Dominant Genres**: Drama (68%), Romance (29%), Action (21%)  
**Key Themes**: Moving relationship stories, Family stories, Heartbreak and suffering

---

### **Cluster 52: Witty Romance and Comic Adventures**
The third-largest cluster: profound human dramas.

**Representative Films**:
- No Other Land (2024)
- The Lord of the Rings: The Return of the King (2003)
- Woman in the Dunes (1964)
- Andrei Rublev (1966)
- Tokyo Story (1953)

**Size**: 132 films (third-largest)  
**Dominant Genres**: Drama (65%), Comedy (25%), Romance (22%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 53: Intimate Stories of the Heart**
Dramas exploring mortality and existential questions.

**Representative Films**:
- Ikiru (1952)
- GoodFellas (1990)
- M (1931)
- Terminator 2: Judgment Day (1991)
- Mulholland Drive (2001)

**Size**: 64 films  
**Dominant Genres**: Drama (73%), Comedy (23%), Crime (22%)  
**Key Themes**: Humanity and the world, Moving relationship stories, Heartbreak and suffering

---

### **Cluster 54: Intimate Portraits of Human Suffering**
Character studies exploring human nature and suffering.

**Representative Films**:
- Sunset Boulevard (1950)
- The Thing (1982)
- Django Unchained (2012)
- Taste of Cherry (1997)
- Man with a Movie Camera (1929)

**Size**: 80 films  
**Dominant Genres**: Drama (69%), Comedy (34%), Romance (18%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 55: Moral Reckoning and Social Justice**
Visually striking dramas with violent and moral themes.

**Representative Films**:
- Lawrence of Arabia (1962)
- Bicycle Thieves (1948)
- The Night of the Hunter (1955)
- The Holdovers (2023)
- Black Swan (2010)

**Size**: 55 films  
**Dominant Genres**: Drama (62%), Comedy (24%), Thriller (22%)  
**Key Themes**: Moving relationship stories, Violence and transgression, Humanity and the world

---

### **Cluster 56: Noir Mysteries and Crime Thrillers**
Another large cluster: diverse dramas and thrillers.

**Representative Films**:
- Close-Up (1990)
- Rear Window (1954)
- The Departed (2006)
- Koyaanisqatsi (1982)
- 13th (2016)

**Size**: 132 films (tied for third-largest)  
**Dominant Genres**: Drama (55%), Comedy (25%), Thriller (24%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Heartbreak and suffering

---

### **Cluster 57: Musical Comedy and Satire**
Intense dramas with surreal and violent elements.

**Representative Films**:
- The Lord of the Rings: The Two Towers (2002)
- WALL·E (2008)
- Drive My Car (2021)
- Dogville (2003)
- Memories of Matsuko (2006)

**Size**: 30 films  
**Dominant Genres**: Drama (67%), Crime (30%), Thriller (27%)  
**Key Themes**: Violence and transgression, Humanity and the world, Surreal visions

---

### **Cluster 58: Heroic Battles Against Darkness**
Relationship dramas with family themes.

**Representative Films**:
- The Dark Knight (2008)
- A Man Escaped (1956)
- Song of the Sea (2014)
- The Fall (2006)
- Aftersun (2022)

**Size**: 62 films  
**Dominant Genres**: Drama (69%), Romance (24%), Thriller (21%)  
**Key Themes**: Moving relationship stories, Humanity and the world, Family stories

---

### **Cluster 59: Political and Historical Documentaries**
Dramas exploring family and suffering.

**Representative Films**:
- Shame (1968)
- The Second Mother (2015)
- Nine Queens (2000)
- Downfall (2004)
- Oasis (2002)

**Size**: 45 films  
**Dominant Genres**: Drama (58%), Comedy (27%), Thriller (27%)  
**Key Themes**: Moving relationship stories, Family stories, Heartbreak and suffering

---

## Methodology

### **Content Feature Extraction**
```
202 total features = 27 genres + 15 themes + 160 filtered tags
```

### **Clustering Algorithm**
```python
kmeans = MiniBatchKMeans(
    n_clusters=60,
    batch_size=10_000,
    random_state=42,
    n_init=20,
    max_iter=500
)
```

### **Manual Labeling**
Cluster names were created independently and don't always reflect algorithmic groupings. The clustering identifies films with similar feature patterns, which sometimes produces unexpected but valid groupings (e.g., "Human Passion and Artistic Struggle" containing both Ran and Das Boot).

---

**Last Updated**: March 2025  
**Dataset**: 4,305 films  
**Clustering**: MiniBatchKMeans (k=60)  
**Features**: 202-dimensional content vectors

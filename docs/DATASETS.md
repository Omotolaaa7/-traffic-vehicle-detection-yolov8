# Jeux de données pour la détection de véhicules en trafic dense

Le projet s'appuie sur des jeux de données publics déjà annotés. Ce document
sert à en choisir un.

Vérifiez la disponibilité et la **licence** de chaque ressource avant usage, et
citez-les dans le rapport.

---

## 1. Le constat de départ

**Il n'existe aucun jeu de données public de vision par ordinateur spécifique au
Bénin** pour la détection de véhicules. Les recherches ne remontent rien, ni sur
Roboflow Universe, ni sur Hugging Face, ni dans la littérature.

C'est une contrainte à énoncer, pas à contourner. Elle a deux conséquences
directes sur ce que vous pouvez affirmer :

- vous ne pouvez pas mesurer l'écart de domaine **béninois**, faute de données
  béninoises pour le mesurer ;
- vous pouvez mesurer, et c'est l'objet du projet, le gain apporté par un
  fine-tuning sur du **trafic urbain dense**, en particulier sur les véhicules
  de petite taille.

Le chiffre qui situe l'enjeu, publié à CVPR 2026 avec le jeu de données
BMD-45 : des détecteurs entraînés sur UA-DETRAC, benchmark occidental classique,
n'atteignent que 33,6 % de mAP@0.50:0.95 sur du trafic urbain dense de pays
émergent, contre 83,8 % lorsqu'ils sont entraînés sur des données du domaine,
soit un facteur 2,5.

Retenez ces deux nombres : ils justifient la démarche en une phrase, et ils
donnent l'ordre de grandeur de l'écart que votre propre comparaison peut
espérer retrouver.

---

## 2. Trafic dense et véhicules de petite taille

### BMD-45 (Hugging Face : `iisc-aim/BMD-45`)

Le jeu de données le plus proche de votre réalité, même s'il est indien.
480 000 boîtes annotées sur 45 000 images issues de plus de 3 600 caméras CCTV
opérationnelles, à Bengaluru. Il comporte 14 catégories fines de véhicules, dont
des modes de transport spécifiques à la région comme les auto-rickshaws.

Pourquoi il concerne directement ce projet : les benchmarks existants
privilégient un trafic homogène et organisé, filmé depuis un véhicule ou en vue
aérienne ; les modèles entraînés sur UA-DETRAC ou COCO généralisent mal aux
conditions denses, hétérogènes et désorganisées des centres urbains en
développement. C'est mot pour mot le problème de Cotonou. Le jeu de données
reproduit les difficultés réelles de déploiement : variation extrême de point de
vue, occlusion, densité de véhicules. Et il est filmé depuis des **caméras
fixes**.

Ne l'utilisez pas intégralement, 45 000 images sont hors budget. **Prenez-en un
sous-ensemble** de 2 000 à 3 000 images pour un pré-affinage, ou servez-vous-en
uniquement comme référence bibliographique.

### Jeu de données routier africain

Publié dans *Engineering, Technology & Applied Science Research*. 3 236 images
originales annotées sur 11 classes : animaux sauvages, animaux domestiques,
véhicules de transport informel (boda-bodas, tuk-tuks, minibus-taxis) et dangers
d'infrastructure. Les **boda-bodas** sont l'équivalent est-africain des
zémidjans. Contactez les auteurs si le lien de téléchargement n'est pas public,
les chercheurs répondent souvent favorablement à des étudiants.

### Jeu de données motos de Kigali

198 images de motos collectées à Kigali, couvrant le trafic congestionné, les
conditions nocturnes et les motos-taxis non conformes, annotées au format COCO
via Roboflow. Trop petit pour entraîner, mais **c'est le modèle exact de ce que
vous devez faire vous-mêmes**, et une publication qui montre qu'un travail à
petite échelle est publiable.

### Benchmarks classiques, pour la comparaison et non l'entraînement

UA-DETRAC, CityFlow, BDD100K, TrafficCAM, VisDrone. Citez-les pour situer le
travail et expliquer pourquoi ils ne servent pas de source principale.

**VisDrone mérite une mention à part** : filmé par drone, il contient une
proportion inhabituelle de très petits véhicules. C'est le benchmark public le
plus proche de la problématique des objets de petite taille, et un point de
comparaison utile si vos propres résultats sur les petits objets sont discutés.

---

## 3. La référence à lire absolument

**Mugizi, Murindanyi, Nakacwa, Katumba, Makerere University (Ouganda)**
*Intelligent Traffic Surveillance for Real-Time Vehicle Detection, License Plate
Recognition, and Speed Estimation*, arxiv.org/html/2601.00344v1

Un projet voisin, réalisé en Afrique de l'Est par une université africaine. Leur
périmètre est plus large que le vôtre, mais leur volet détection donne un repère
chiffré crédible : mAP de 97,9 % avec YOLOv8 sur leur propre jeu de données.

Citer ce papier en soutenance et montrer que vous connaissez l'état de l'art
**africain** sur le sujet vous distingue immédiatement.

---

## 4. Comment choisir

Ne cherchez pas le jeu parfait, il n'existe pas. Cherchez celui qui contient le
plus de **petits objets** et de **scènes denses**, puisque c'est ce que le projet
prétend améliorer.

### Les quatre critères, par ordre d'importance

| Critère | Pourquoi | Comment vérifier en 5 minutes |
|---|---|---|
| Beaucoup de véhicules éloignés et petits | C'est l'objet même de l'étude | Ouvrez une dizaine d'images : voit-on des véhicules au fond ? |
| Trafic dense, véhicules qui se chevauchent | Second axe du sujet | Cherchez des scènes d'embouteillage réel |
| Prise de vue fixe, en hauteur | Correspond au cas d'usage de surveillance | Vue CCTV plutôt que vue depuis un véhicule |
| Classes proches de COCO | Condition de validité de la comparaison | Voiture, moto, bus, camion doivent exister |

Le quatrième critère est le plus contraignant en pratique. Un jeu dont les
classes sont `rickshaw`, `tuk-tuk`, `van`, `pickup` demandera beaucoup
d'arbitrages de correspondance, et chaque classe sans équivalent COCO devra être
ignorée, ce qui réduit d'autant les données exploitables.

### Un piège fréquent, et un bon sujet de soutenance

Certains jeux publics présentent une **contamination entre entraînement et
test** : les mêmes images sources apparaissent dans plusieurs splits, parfois
avec une simple augmentation. Les métriques publiées dessus sont alors
surestimées.

Vérifiez-le sur le jeu que vous retenez : ouvrez quelques images de test et
cherchez leurs quasi-doublons dans l'entraînement. Si vous en trouvez, dites-le
et refaites le découpage. Repérer ce genre de problème et le signaler est
exactement ce qu'un jury technique valorise, bien plus qu'un chiffre élevé.

### Volumétrie utile

| Taille du jeu | Ce que vous pouvez en faire |
|---|---|
| Moins de 500 images | Insuffisant, les métriques ne seront pas interprétables |
| 1 000 à 3 000 images | Suffisant pour un fine-tuning démonstratif, c'est la cible |
| Plus de 10 000 images | Prenez-en un sous-ensemble, le temps machine deviendrait prohibitif |

### Le protocole qui fera votre soutenance

Une seule expérience, et elle est décisive :

1. Importez le jeu retenu, en vous assurant d'un split de test isolé.
2. Évaluez le modèle pré-entraîné sur ce jeu de test.
3. Fine-tunez, puis évaluez le nouveau modèle sur **le même** jeu de test.
4. Reportez l'écart, en distinguant les objets petits, moyens et grands.

C'est exactement ce qu'implémente `scripts/compare_models.py`, et c'est
méthodologiquement irréprochable à condition que le point 1 soit respecté.

La ventilation par taille du point 4 est ce qui rattache vos chiffres au titre
du rapport. Sans elle, vous mesurez une amélioration générique ; avec elle, vous
répondez à la question posée.

---

## 5. Licences et citation

- **Vérifiez la licence de chaque jeu de données** et listez-les dans le
  rapport. Les datasets Roboflow Universe portent des licences hétérogènes,
  souvent CC BY 4.0, parfois non spécifiée. Une licence non spécifiée n'est pas
  une autorisation.
- **Citez chaque source au format BibTeX** fourni par la plateforme.
- Si vous illustrez vos slides avec des images du jeu, vérifiez que la licence
  autorise la reproduction, et floutez plaques et visages par précaution.

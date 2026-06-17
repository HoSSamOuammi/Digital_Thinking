from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "rapport_technique"
SCREEN_DIR = OUT_DIR / "screenshots"
APP_SCREEN_DIR = SCREEN_DIR / "application"
GITHUB_SCREEN_DIR = SCREEN_DIR / "github"
PDF_PATH = OUT_DIR / "rapport_technique.pdf"
MD_PATH = OUT_DIR / "rapport_technique.md"

BASE_COMMIT = "542daf3"
REPO_URL = "https://github.com/HoSSamOuammi/Digital_Thinking"

TEAM = [
    {
        "name": "Hossam OUammi",
        "email": "Ouammi.hossam@etu.uae.ac.ma",
        "role": "Intégration Flask, design administratif, module médias, galerie et pages de présentation.",
    },
    {
        "name": "Aya EL Amrani",
        "email": "ElAamrani.aya@etu.uae.ac.ma",
        "role": "Architecture applicative, configuration, formulaires, stockage, sécurité et routes principales.",
    },
    {
        "name": "Khadija Baskar",
        "email": "Baskar.Khadija@etu.uae.ac.ma",
        "role": "Traduction française, libellés, textes visibles et cohérence des intitulés.",
    },
    {
        "name": "Abderrahmane El Garti",
        "email": "ElGarti.abderrahmane@etu.uae.ac.ma",
        "role": "Tests fonctionnels, documentation, rapport et analyse technique.",
    },
]

FEATURES = [
    {
        "title": "Tableau de bord",
        "body": (
            "La page d’accueil donne une vue rapide sur l’état du studio : nombre d’images générées, "
            "fichiers audio disponibles, palettes intégrées et accès directs vers les ateliers. "
            "Elle sert de point d’entrée pour présenter le projet sans obliger l’utilisateur à connaître la structure interne."
        ),
        "image": APP_SCREEN_DIR / "01_accueil.png",
    },
    {
        "title": "Atelier génératif",
        "body": (
            "L’atelier génératif permet de produire des visuels à partir de paramètres contrôlés : série visuelle, "
            "fond, palette, graine, nombre de formes, densité, taille du canevas et dessin d’accents sur l’aperçu. "
            "La prévisualisation donne un retour rapide avant l’export final."
        ),
        "image": APP_SCREEN_DIR / "02_atelier_generatif.png",
    },
    {
        "title": "Visualisation de données",
        "body": (
            "Le module de données accepte un fichier CSV ou utilise un jeu de données de démonstration. "
            "Les colonnes numériques sont nettoyées, lissées et transformées en visuels : vue complète, paysage, "
            "carte thermique, barres graduées ou rayonnement circulaire."
        ),
        "image": APP_SCREEN_DIR / "03_donnees_visuelles.png",
    },
    {
        "title": "Outils médias",
        "body": (
            "La partie médias permet de traiter une image avec plusieurs effets : noir et blanc, sépia, inversion, "
            "flou, contours, pixelisation, miroir, rotation, glitch, aquarelle et palette dominante. "
            "Le traitement audio est prévu quand l’environnement dispose de PyDub et ffmpeg."
        ),
        "image": APP_SCREEN_DIR / "04_outils_medias.png",
    },
    {
        "title": "Galerie",
        "body": (
            "La galerie regroupe les fichiers générés dans l’application. Les images et fichiers audio sont listés "
            "séparément, avec pagination et liens de téléchargement. Cette page sert aussi de preuve visuelle du flux complet : "
            "génération, sauvegarde, consultation et export."
        ),
        "image": APP_SCREEN_DIR / "05_galerie.png",
    },
    {
        "title": "Équipe",
        "body": (
            "La page équipe présente les membres, leurs rôles et leurs emails. Elle relie la partie fonctionnelle du projet "
            "à la répartition du travail visible dans l’historique Git."
        ),
        "image": APP_SCREEN_DIR / "06_equipe.png",
    },
]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def get_commits() -> list[dict[str, str]]:
    raw = run_git(
        [
            "log",
            "--reverse",
            "--format=%h%x1f%an%x1f%ae%x1f%ad%x1f%s",
            "--date=short",
            f"{BASE_COMMIT}..HEAD",
        ]
    )
    commits = []
    for line in raw.splitlines():
        commit_hash, author, email, commit_date, subject = line.split("\x1f")
        commits.append(
            {
                "hash": commit_hash,
                "author": author,
                "email": email,
                "date": commit_date,
                "subject": subject,
            }
        )
    return commits


def get_commit_counts(commits: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for commit in commits:
        counts[commit["author"]] += 1
    return dict(counts)


def commit_period(commits: list[dict[str, str]]) -> str:
    dates = [commit["date"] for commit in commits]
    return f"{min(dates)} au {max(dates)}"


def grouped_commits(commits: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for commit in commits:
        grouped[commit["author"]].append(commit)
    return grouped


def build_markdown(commits: list[dict[str, str]], counts: dict[str, int]) -> None:
    grouped = grouped_commits(commits)
    lines = [
        "# Rapport technique",
        "",
        "## Projet",
        "",
        "Studio génératif interactif est une application Flask qui regroupe plusieurs ateliers numériques.",
        "L’utilisateur peut générer des images, transformer des données, traiter des fichiers médias et consulter les exports dans une galerie.",
        "",
        "## Fonctionnalités",
        "",
    ]
    for feature in FEATURES:
        lines.append(f"### {feature['title']}")
        lines.append("")
        lines.append(feature["body"])
        lines.append("")

    lines.extend(
        [
            "## Architecture",
            "",
            "- `app.py` : lancement de l’application.",
            "- `studio/app_factory.py` : création Flask et enregistrement des routes.",
            "- `studio/routes/` : pages séparées par fonctionnalité.",
            "- `studio/forms.py` : lecture et validation des formulaires.",
            "- `studio/storage.py` : gestion des fichiers.",
            "- `modules/` : logique métier.",
            "",
            "## Répartition",
            "",
            "| Membre | Commits | Partie principale |",
            "| --- | ---: | --- |",
        ]
    )
    for member in TEAM:
        lines.append(f"| {member['name']} | {counts.get(member['name'], 0)} | {member['role']} |")

    lines.extend(
        [
            "",
            "## Suivi Git",
            "",
            f"- Dépôt : {REPO_URL}",
            f"- Période des commits : {commit_period(commits)}",
            f"- Total : {len(commits)} commits",
            "",
        ]
    )
    for member in TEAM:
        lines.append(f"### {member['name']}")
        for commit in grouped.get(member["name"], []):
            lines.append(f"- {commit['date']} - `{commit['hash']}` - {commit['subject']}")
        lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleAdmin",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#202124"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#202124"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitleAdmin",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#202124"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#3c4043"),
        )
    )
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.2
    styles["BodyText"].leading = 12.5
    styles["BodyText"].textColor = colors.HexColor("#202124")
    return styles


STYLES = make_styles()


def table(data, widths):
    output = Table(data, colWidths=widths, repeatRows=1)
    output.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f3f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#202124")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return output


def bullet_list(items: list[str]):
    return ListFlowable(
        [ListItem(Paragraph(item, STYLES["BodyText"])) for item in items],
        bulletType="bullet",
        leftIndent=14,
    )


def screenshot(path: Path, caption: str, max_width: float = 17.2 * cm):
    if not path.exists():
        return []
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = max_width
    image.drawHeight = max_width * ratio
    return [
        Paragraph(caption, STYLES["SmallText"]),
        Spacer(1, 0.12 * cm),
        image,
        Spacer(1, 0.45 * cm),
    ]


def sample_commits(commits: list[dict[str, str]], author: str, limit: int = 6) -> list[dict[str, str]]:
    return [commit for commit in commits if commit["author"] == author][:limit]


def build_pdf(commits: list[dict[str, str]], counts: dict[str, int]) -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.45 * cm,
        title="Rapport technique - Studio génératif interactif",
    )
    story = []

    story.append(Paragraph("Rapport technique", STYLES["TitleAdmin"]))
    story.append(Paragraph("Studio génératif interactif", STYLES["SubTitleAdmin"]))
    story.append(Paragraph("Module : Créativité numérique", STYLES["BodyText"]))
    story.append(Paragraph(f"Dépôt GitHub : {REPO_URL}", STYLES["BodyText"]))
    story.append(Paragraph(f"Date du rapport : {date.today().strftime('%d/%m/%Y')}", STYLES["BodyText"]))
    story.append(Spacer(1, 0.9 * cm))

    story.append(Paragraph("1. Présentation du projet", STYLES["SectionTitle"]))
    story.append(
        Paragraph(
            "Le projet consiste à créer une application web Flask pour produire et manipuler des contenus numériques. "
            "L’application rassemble plusieurs ateliers dans une interface unique : génération de visuels, visualisation "
            "de données, traitement d’images, traitement audio optionnel et galerie des résultats exportés.",
            STYLES["BodyText"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            "Le travail a été organisé autour d’une structure simple afin de rester explicable : les routes Flask sont "
            "séparées, les formulaires sont centralisés, la gestion des fichiers est isolée et les traitements principaux "
            "restent dans le dossier modules.",
            STYLES["BodyText"],
        )
    )

    story.append(Paragraph("2. Ce que l’utilisateur peut faire", STYLES["SectionTitle"]))
    story.append(
        bullet_list(
            [
                "Créer des visuels génératifs avec des paramètres ajustables.",
                "Importer ou utiliser un jeu de données de démonstration pour produire une visualisation.",
                "Appliquer des effets à une image et télécharger le résultat.",
                "Utiliser les outils audio si ffmpeg est disponible sur la machine.",
                "Consulter les fichiers générés dans une galerie avec pagination.",
                "Retrouver les informations de l’équipe et les rôles de chacun.",
            ]
        )
    )

    story.append(Paragraph("3. Technologies utilisées", STYLES["SectionTitle"]))
    story.append(
        table(
            [
                ["Technologie", "Utilisation dans le projet"],
                ["Flask", "Routes, formulaires, rendu des pages HTML."],
                ["Jinja2", "Templates dynamiques côté serveur."],
                ["Matplotlib / NumPy", "Génération de visuels et visualisation de données."],
                ["Pandas", "Lecture et préparation des données CSV."],
                ["Pillow", "Traitements d’image."],
                ["PyDub", "Traitements audio lorsque l’environnement le permet."],
                ["unittest", "Tests fonctionnels des routes principales."],
            ],
            [4.2 * cm, 12.6 * cm],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Fonctionnalités de l’application", STYLES["SectionTitle"]))
    for index, feature in enumerate(FEATURES, start=1):
        story.append(Paragraph(f"4.{index} {feature['title']}", STYLES["SubTitleAdmin"]))
        story.append(Paragraph(feature["body"], STYLES["BodyText"]))
        story.extend(screenshot(feature["image"], f"Capture réelle - {feature['title']}"))
        if index in {2, 4}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("5. Architecture technique", STYLES["SectionTitle"]))
    story.append(
        Paragraph(
            "L’application est organisée en couches simples. Cette organisation rend le code plus facile à lire, à tester "
            "et à présenter. Le fichier app.py ne contient plus la logique complète ; il sert uniquement à lancer l’application.",
            STYLES["BodyText"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        table(
            [
                ["Élément", "Rôle"],
                ["app.py", "Point d’entrée de l’application."],
                ["studio/app_factory.py", "Création de l’application Flask et enregistrement des routes."],
                ["studio/routes/", "Routes séparées par page ou atelier."],
                ["studio/forms.py", "Lecture, conversion et validation simple des paramètres."],
                ["studio/storage.py", "Sauvegarde, pagination et nettoyage des fichiers."],
                ["studio/security.py", "Protection CSRF des formulaires."],
                ["studio/labels.py", "Libellés français affichés dans l’interface."],
                ["modules/", "Fonctions métier : génération, données, image, audio."],
                ["templates/", "Pages HTML Jinja2."],
                ["static/", "CSS, images, fichiers générés et fichiers importés."],
            ],
            [4.2 * cm, 12.6 * cm],
        )
    )

    story.append(Paragraph("6. Parcours de fonctionnement", STYLES["SectionTitle"]))
    story.append(
        bullet_list(
            [
                "L’utilisateur ouvre une page d’atelier depuis le tableau de bord.",
                "Il remplit un formulaire ou modifie des paramètres.",
                "Flask lit et sécurise les données reçues.",
                "Le module Python correspondant génère ou transforme un fichier.",
                "Le résultat est sauvegardé dans static/generated.",
                "La page affiche le résultat et propose le téléchargement.",
            ]
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Répartition du travail", STYLES["SectionTitle"]))
    story.append(
        table(
            [["Membre", "Commits", "Partie principale"]]
            + [[m["name"], str(counts.get(m["name"], 0)), m["role"]] for m in TEAM],
            [4.4 * cm, 2.0 * cm, 10.4 * cm],
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    for member in TEAM:
        rows = [["Date", "Message"]]
        for commit in sample_commits(commits, member["name"]):
            rows.append([commit["date"], commit["subject"]])
        block = [
            Paragraph(member["name"], STYLES["SubTitleAdmin"]),
            Paragraph(member["role"], STYLES["BodyText"]),
            Spacer(1, 0.12 * cm),
            table(rows, [2.8 * cm, 13.9 * cm]),
            Spacer(1, 0.35 * cm),
        ]
        story.append(KeepTogether(block))

    story.append(PageBreak())
    story.append(Paragraph("8. Validation", STYLES["SectionTitle"]))
    story.append(
        table(
            [
                ["Élément vérifié", "Résultat"],
                ["Tests", "13 tests OK avec python -m unittest discover -s tests -v"],
                ["Pages principales", "Accueil, équipe, génératif, données, médias et galerie rendues correctement."],
                ["Exports", "Images générées et consultables dans la galerie."],
                ["Historique Git", f"{len(commits)} commits entre {commit_period(commits)}."],
            ],
            [4.8 * cm, 12.0 * cm],
        )
    )

    story.append(Paragraph("9. Suivi GitHub", STYLES["SectionTitle"]))
    story.append(
        Paragraph(
            "La partie GitHub sert de trace de travail : historique des commits, aperçu Pulse et répartition des contributeurs. "
            "Les captures suivantes ont été prises après la mise à jour de la branche main.",
            STYLES["BodyText"],
        )
    )
    story.extend(screenshot(GITHUB_SCREEN_DIR / "02_commits.png", "Capture GitHub - historique des commits"))
    story.append(PageBreak())
    story.extend(screenshot(GITHUB_SCREEN_DIR / "03_pulse.png", "Capture GitHub - Pulse mensuel"))
    story.extend(screenshot(GITHUB_SCREEN_DIR / "04_contributors.png", "Capture GitHub - Contributors"))

    story.append(PageBreak())
    story.append(Paragraph("Annexe - liste des commits", STYLES["SectionTitle"]))
    grouped = grouped_commits(commits)
    for member in TEAM:
        story.append(Paragraph(member["name"], STYLES["SubTitleAdmin"]))
        rows = [["Date", "Hash", "Message"]]
        for commit in grouped.get(member["name"], []):
            rows.append([commit["date"], commit["hash"], commit["subject"]])
        story.append(table(rows, [2.4 * cm, 2.0 * cm, 12.3 * cm]))
        story.append(Spacer(1, 0.35 * cm))

    doc.build(story)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commits_data = get_commits()
    counts_data = get_commit_counts(commits_data)
    build_markdown(commits_data, counts_data)
    build_pdf(commits_data, counts_data)
    print(PDF_PATH)
    print(MD_PATH)

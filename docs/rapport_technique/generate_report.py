from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import date
from html import escape
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
ADMINS_DIR = ROOT / "static" / "Admins"

BASE_COMMIT = "542daf3"
REPO_URL = "https://github.com/HoSSamOuammi/Digital_Thinking"
PROJECT_START = "04/03/2026"
TEAM_IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp")

TEAM = [
    {
        "slug": "aya",
        "name": "Aya EL Amrani",
        "email": "ElAamrani.aya@etu.uae.ac.ma",
        "role": "Structure Flask, configuration, formulaires, stockage et routes.",
    },
    {
        "slug": "khadija",
        "name": "Khadija Baskar",
        "email": "Baskar.Khadija@etu.uae.ac.ma",
        "role": "Textes français, libellés, cohérence des intitulés et contenu des pages.",
    },
    {
        "slug": "hossam",
        "name": "Hossam OUammi",
        "email": "Ouammi.hossam@etu.uae.ac.ma",
        "role": "Intégration Flask, interface, médias, galerie et pages de présentation.",
    },
    {
        "slug": "abdo",
        "name": "Abderrahmane El Garti",
        "email": "ElGarti.abderrahmane@etu.uae.ac.ma",
        "role": "Tests fonctionnels, documentation, rapport et analyse technique.",
    },
]

AUTHOR_ALIASES = {
    "Hossam Ouammi": "Hossam OUammi",
}

FEATURES = [
    {
        "title": "Tableau de bord",
        "body": (
            "La page d'accueil donne le ton du projet. On y retrouve les compteurs, les accès aux ateliers "
            "et les derniers exports. C'est volontairement simple: quelqu'un qui découvre le projet doit comprendre "
            "en quelques secondes où cliquer pour tester l'application."
        ),
        "image": APP_SCREEN_DIR / "01_accueil.png",
    },
    {
        "title": "Atelier génératif",
        "body": (
            "L'atelier génératif est la partie la plus créative. L'utilisateur règle une série visuelle, une palette, "
            "un fond, une graine et plusieurs paramètres de densité ou de taille. La prévisualisation aide à tester "
            "rapidement une idée avant de lancer l'export final."
        ),
        "image": APP_SCREEN_DIR / "02_atelier_generatif.png",
    },
    {
        "title": "Données visuelles",
        "body": (
            "Ce module transforme un CSV, ou un jeu de démonstration, en image. Les valeurs numériques sont nettoyées "
            "puis utilisées pour produire une visualisation. L'intérêt est de montrer que le studio ne génère pas "
            "seulement des formes abstraites: il peut aussi partir de données."
        ),
        "image": APP_SCREEN_DIR / "03_donnees_visuelles.png",
    },
    {
        "title": "Outils médias",
        "body": (
            "La page médias permet d'importer une image et d'appliquer des effets visibles: sépia, contours, rotation, "
            "glitch, palette dominante, etc. Le traitement audio reste prévu, mais il dépend de ffmpeg, donc l'application "
            "affiche clairement l'état de disponibilité au lieu de planter."
        ),
        "image": APP_SCREEN_DIR / "04_outils_medias.png",
    },
    {
        "title": "Galerie",
        "body": (
            "La galerie ferme le parcours utilisateur. Après une génération ou un traitement, les fichiers sont listés "
            "avec pagination et liens de téléchargement. Cette page prouve que le flux complet fonctionne: créer, sauvegarder, "
            "retrouver, télécharger."
        ),
        "image": APP_SCREEN_DIR / "05_galerie.png",
    },
    {
        "title": "Équipe",
        "body": (
            "La page équipe présente les membres, leurs rôles, leurs emails et leurs photos. Elle rend le projet plus humain "
            "et permet de relier les parties techniques à la répartition réelle du travail."
        ),
        "image": APP_SCREEN_DIR / "06_equipe.png",
    },
]

TOOLS = [
    ["Flask", "Routage, formulaires, sessions et rendu des pages."],
    ["Jinja2", "Templates HTML avec données envoyées par Flask."],
    ["Pillow", "Lecture, transformation et export d'images."],
    ["Pandas / NumPy", "Préparation des données CSV et calculs numériques."],
    ["Matplotlib", "Création des visualisations exportées en image."],
    ["PyDub / ffmpeg", "Traitement audio lorsque l'environnement le permet."],
    ["unittest", "Tests fonctionnels des pages et traitements principaux."],
    ["Git / GitHub", "Historique de travail, dépôt final et preuves de collaboration."],
]

PIPELINE = [
    "L'utilisateur choisit un atelier depuis le tableau de bord.",
    "Le formulaire est envoyé à Flask avec un jeton CSRF.",
    "Les paramètres sont lus et normalisés dans studio/forms.py.",
    "Le module métier correspondant génère ou transforme le contenu.",
    "Le fichier obtenu est sauvegardé dans static/generated.",
    "La page affiche le résultat et propose le téléchargement.",
    "Les tests rejouent les parcours importants pour éviter les régressions.",
]

CHALLENGES = [
    [
        "Le fichier app.py était trop chargé",
        "La logique a été déplacée vers une fabrique Flask, des fichiers de routes et des services simples. Le projet est plus facile à lire et à expliquer.",
    ],
    [
        "L'interface devait paraître terminée",
        "Les textes ont été harmonisés en français et le design a été rendu plus sobre pour ressembler à un vrai outil étudiant.",
    ],
    [
        "Les exports pouvaient encombrer le dossier static",
        "Le stockage a été isolé, les imports temporaires sont supprimés et la galerie utilise une pagination.",
    ],
    [
        "Le module audio dépend de ffmpeg",
        "L'application détecte l'état de l'audio et reste utilisable même quand ffmpeg n'est pas disponible.",
    ],
    [
        "Les tests devaient suivre la nouvelle architecture",
        "Les chemins de patch et les assertions ont été adaptés après la séparation des routes.",
    ],
]


def report_date() -> str:
    return date.today().strftime("%d/%m/%Y")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def canonical_author(author: str) -> str:
    return AUTHOR_ALIASES.get(author, author)


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
    commits: list[dict[str, str]] = []
    for line in raw.splitlines():
        commit_hash, author, email, commit_date, subject = line.split("\x1f")
        commits.append(
            {
                "hash": commit_hash,
                "author": canonical_author(author),
                "email": email,
                "date": commit_date,
                "subject": subject,
            }
        )
    return commits


def commit_period(commits: list[dict[str, str]]) -> str:
    dates = [commit["date"] for commit in commits]
    return f"{min(dates)} au {max(dates)}" if dates else "non disponible"


def grouped_commits(commits: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for commit in commits:
        grouped[commit["author"]].append(commit)
    return grouped


def sample_commits(commits: list[dict[str, str]], author: str, limit: int = 5) -> list[dict[str, str]]:
    return [commit for commit in commits if commit["author"] == author][:limit]


def team_photo_path(member: dict[str, str]) -> Path | None:
    for extension in TEAM_IMAGE_EXTENSIONS:
        candidate = ADMINS_DIR / f"{member['slug']}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def markdown_photo(member: dict[str, str]) -> str:
    photo_path = team_photo_path(member)
    if not photo_path:
        return "-"
    return f"![{member['name']}](../../static/Admins/{photo_path.name})"


def build_markdown(commits: list[dict[str, str]]) -> None:
    grouped = grouped_commits(commits)
    lines = [
        "# Rapport technique détaillé - Studio génératif interactif",
        "",
        f"- Date du rapport : {report_date()}",
        f"- Période Git observée : {commit_period(commits)}",
        f"- Dépôt GitHub : {REPO_URL}",
        "",
        "## 1. Idée générale et direction artistique",
        "",
        "Le projet prend la forme d'un petit studio numérique. L'utilisateur peut générer des images, transformer des données, "
        "appliquer des effets médias puis retrouver les exports dans une galerie. Nous avons voulu garder une interface claire, "
        "avec un style sobre et des textes français, pour que le projet soit agréable à présenter et facile à tester.",
        "",
        "## 2. Modules réalisés",
        "",
    ]

    for feature in FEATURES:
        lines.append(f"### {feature['title']}")
        lines.append("")
        lines.append(feature["body"])
        lines.append("")

    lines.extend(
        [
            "## 3. Architecture et pipeline",
            "",
            "- `app.py` lance l'application.",
            "- `studio/app_factory.py` crée Flask et enregistre les routes.",
            "- `studio/routes/` sépare les vues par domaine.",
            "- `studio/forms.py`, `storage.py` et `security.py` isolent les tâches répétitives.",
            "- `modules/` contient les traitements de génération, données, image et audio.",
            "",
            "Pipeline :",
        ]
    )
    for step in PIPELINE:
        lines.append(f"- {step}")

    lines.extend(["", "## 4. Outils utilisés", "", "| Outil | Utilisation |", "| --- | --- |"])
    for tool, usage in TOOLS:
        lines.append(f"| {tool} | {usage} |")

    lines.extend(["", "## 5. Challenges et solutions", "", "| Challenge | Solution |", "| --- | --- |"])
    for challenge, solution in CHALLENGES:
        lines.append(f"| {challenge} | {solution} |")

    lines.extend(["", "## 6. Équipe", "", "| Photo | Membre | Rôle |", "| --- | --- | --- |"])
    for member in TEAM:
        lines.append(f"| {markdown_photo(member)} | {member['name']} | {member['role']} |")

    lines.extend(["", "## 7. Extraits du suivi Git", ""])
    for member in TEAM:
        lines.append(f"### {member['name']}")
        for commit in sample_commits(commits, member["name"]):
            lines.append(f"- {commit['date']} - `{commit['hash']}` - {commit['subject']}")
        if not grouped.get(member["name"]):
            lines.append("- Aucun commit retrouvé avec ce nom d'auteur.")
        lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            textColor=colors.HexColor("#17324d"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#1f5f8f"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.2,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.2,
            textColor=colors.HexColor("#3c4043"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=8.8,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.6,
            leading=9,
            textColor=colors.HexColor("#17324d"),
        )
    )
    return styles


STYLES = make_styles()


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(escape(text), STYLES[style])


def cell(value, header: bool = False):
    if hasattr(value, "wrap"):
        return value
    return Paragraph(escape(str(value)), STYLES["TableHead" if header else "TableText"])


def table(data, widths):
    rows = [[cell(value, header=index == 0) for value in row] for index, row in enumerate(data)]
    output = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    output.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4f8")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8e3")),
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
        [ListItem(p(item, "Body"), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=16,
        bulletFontSize=6,
    )


def scaled_image(path: Path, max_width: float, max_height: float | None = None):
    image = Image(str(path))
    max_height = max_height or 1000 * cm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def screenshot(path: Path, caption: str, max_width: float = 17.2 * cm, max_height: float = 9.6 * cm):
    if not path.exists():
        return []
    return [
        Paragraph(escape(caption), STYLES["Small"]),
        Spacer(1, 0.1 * cm),
        scaled_image(path, max_width, max_height),
        Spacer(1, 0.35 * cm),
    ]


def member_photo(member: dict[str, str]):
    photo_path = team_photo_path(member)
    if not photo_path:
        return p("-", "TableText")
    return scaled_image(photo_path, 1.6 * cm, 2.0 * cm)


def build_pdf(commits: list[dict[str, str]]) -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.25 * cm,
        title="Rapport technique détaillé - Studio génératif interactif",
    )
    story = []

    story.append(Paragraph("Rapport technique détaillé", STYLES["ReportTitle"]))
    story.append(p("Studio génératif interactif - projet Flask de créativité numérique"))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        table(
            [
                ["Élément", "Information"],
                ["Date du rapport", report_date()],
                ["Période Git observée", commit_period(commits)],
                ["Dépôt GitHub", REPO_URL],
                ["Objectif", "Présenter l'application, ses modules, son pipeline technique et les choix faits pendant le projet."],
            ],
            [4.1 * cm, 13.0 * cm],
        )
    )
    story.append(Spacer(1, 0.45 * cm))

    story.append(Paragraph("1. Concept et direction artistique", STYLES["Section"]))
    story.append(
        p(
            "Nous avons pensé l'application comme un petit studio numérique. L'utilisateur peut partir de réglages simples, "
            "d'un fichier CSV ou d'une image, produire un rendu, puis le retrouver dans une galerie. Le projet reste volontairement "
            "accessible: il doit se tester rapidement et se comprendre sans lire tout le code."
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        p(
            "La direction visuelle est sobre: fond clair, cartes lisibles, navigation stable et textes français. Ce choix donne "
            "une impression d'outil terminé, sans transformer le projet en page promotionnelle trop chargée."
        )
    )

    story.append(Paragraph("2. Vue d'ensemble des modules", STYLES["Section"]))
    story.append(table([["Module", "Rôle dans le projet"]] + [[f["title"], f["body"]] for f in FEATURES], [4.0 * cm, 13.1 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("3. Parcours illustré de l'application", STYLES["Section"]))
    for index, feature in enumerate(FEATURES, start=1):
        story.append(Paragraph(f"3.{index} {feature['title']}", STYLES["Subsection"]))
        story.append(p(feature["body"]))
        story.extend(screenshot(feature["image"], f"Capture de l'application - {feature['title']}"))
        if index in {2, 4}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("4. Architecture et pipeline technique", STYLES["Section"]))
    story.append(
        p(
            "Le projet a été séparé pour éviter un seul fichier trop long. `app.py` lance l'application, `studio/app_factory.py` "
            "prépare Flask, `studio/routes/` contient les vues, et `modules/` garde les traitements métier. Cette structure reste "
            "simple, mais elle donne une vraie logique au projet."
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        table(
            [
                ["Fichier ou dossier", "Rôle"],
                ["app.py", "Point d'entrée de l'application."],
                ["studio/app_factory.py", "Création de Flask et enregistrement des routes."],
                ["studio/routes/", "Pages séparées par fonctionnalité."],
                ["studio/forms.py", "Lecture et validation des paramètres."],
                ["studio/storage.py", "Sauvegarde, pagination et nettoyage des fichiers."],
                ["studio/security.py", "Protection CSRF des formulaires."],
                ["modules/", "Génération artistique, données, image et audio."],
                ["templates/", "Pages HTML Jinja2."],
                ["static/", "CSS, photos, captures, exports et fichiers générés."],
            ],
            [4.6 * cm, 12.5 * cm],
        )
    )
    story.append(Paragraph("Pipeline de fonctionnement", STYLES["Subsection"]))
    story.append(bullet_list(PIPELINE))

    story.append(Paragraph("5. Outils utilisés", STYLES["Section"]))
    story.append(table([["Outil", "Utilisation"]] + TOOLS, [4.2 * cm, 12.9 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("6. Challenges et solutions", STYLES["Section"]))
    story.append(
        p(
            "Les difficultés principales n'étaient pas seulement techniques. Il fallait aussi rendre le projet lisible, présentable "
            "et assez stable pour être lancé pendant une soutenance."
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(table([["Challenge", "Solution"]] + CHALLENGES, [5.4 * cm, 11.7 * cm]))

    story.append(Paragraph("7. Équipe et répartition du travail", STYLES["Section"]))
    story.append(
        table(
            [["Photo", "Membre", "Rôle"]]
            + [[member_photo(member), member["name"], member["role"]] for member in TEAM],
            [2.0 * cm, 4.2 * cm, 10.9 * cm],
        )
    )

    story.append(Spacer(1, 0.25 * cm))
    for member in TEAM:
        rows = [["Date", "Commit", "Message"]]
        for commit in sample_commits(commits, member["name"]):
            rows.append([commit["date"], commit["hash"], commit["subject"]])
        story.append(
            KeepTogether(
                [
                    Paragraph(member["name"], STYLES["Subsection"]),
                    table(rows, [2.2 * cm, 2.0 * cm, 12.9 * cm]),
                    Spacer(1, 0.25 * cm),
                ]
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("8. Validation", STYLES["Section"]))
    story.append(
        table(
            [
                ["Point vérifié", "Résultat"],
                ["Tests", "La suite unittest couvre les routes principales, la génération, les formulaires protégés et le nettoyage."],
                ["Application", "Les pages principales se chargent correctement: accueil, équipe, génératif, données, médias et galerie."],
                ["Exports", "Les fichiers générés sont sauvegardés dans static/generated puis visibles dans la galerie."],
                ["Dépôt", "Le dépôt GitHub contient le code, le README, le rapport, les images d'équipe et les captures utilisées."],
            ],
            [4.2 * cm, 12.9 * cm],
        )
    )

    story.append(Paragraph("9. Suivi GitHub", STYLES["Section"]))
    story.append(
        p(
            "Les captures GitHub montrent le dépôt, l'historique et l'activité du projet. Elles servent surtout de preuve visuelle "
            "du suivi et de la présence des livrables dans le dépôt."
        )
    )
    story.extend(screenshot(GITHUB_SCREEN_DIR / "01_depot.png", "Capture GitHub - dépôt"))
    story.extend(screenshot(GITHUB_SCREEN_DIR / "02_commits.png", "Capture GitHub - historique des commits"))
    story.append(PageBreak())
    story.extend(screenshot(GITHUB_SCREEN_DIR / "03_pulse.png", "Capture GitHub - activité Pulse"))

    story.append(Paragraph("10. Conclusion", STYLES["Section"]))
    story.append(
        p(
            "La version finale reste simple, mais c'est justement son intérêt. Elle montre une application Flask complète avec "
            "plusieurs modules, une interface cohérente, des exports, une galerie, des tests et un dépôt propre. Le projet est donc "
            "présentable autant côté code que côté usage."
        )
    )

    doc.build(story)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commits_data = get_commits()
    build_markdown(commits_data)
    build_pdf(commits_data)
    print(PDF_PATH)
    print(MD_PATH)

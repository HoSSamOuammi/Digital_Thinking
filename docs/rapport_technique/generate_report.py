from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
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
PDF_PATH = OUT_DIR / "rapport_technique.pdf"
MD_PATH = OUT_DIR / "rapport_technique.md"
ADMINS_DIR = ROOT / "static" / "Admins"

REPO_URL = "https://github.com/HoSSamOuammi/Digital_Thinking"
REPORT_DATE = date.today()
PROJECT_START = "04/03/2026"
TEAM_IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp")

TEAM = [
    {
        "slug": "aya",
        "name": "Aya EL Amrani",
        "role": "Structure Flask, configuration, formulaires, stockage et routes.",
    },
    {
        "slug": "khadija",
        "name": "Khadija Baskar",
        "role": "Textes français, libellés, cohérence des intitulés et contenu des pages.",
    },
    {
        "slug": "hossam",
        "name": "Hossam OUammi",
        "role": "Intégration Flask, interface, médias, galerie et pages de présentation.",
    },
    {
        "slug": "abdo",
        "name": "Abderrahmane El Garti",
        "role": "Tests fonctionnels, documentation, rapport et analyse technique.",
    },
]

MODULES = [
    [
        "Tableau de bord",
        "Accueil du studio, compteurs et raccourcis vers les ateliers.",
        "Une entrée simple pour présenter le projet rapidement.",
    ],
    [
        "Atelier génératif",
        "Séries visuelles, palettes, graine, densité, taille du canevas et accents dessinés.",
        "Visuels exportés puis visibles dans la galerie.",
    ],
    [
        "Données visuelles",
        "Lecture d'un CSV ou d'un jeu de démonstration, nettoyage puis rendu graphique.",
        "Images de visualisation prêtes à télécharger.",
    ],
    [
        "Outils médias",
        "Effets image: sépia, contours, glitch, rotation, palette dominante. Audio si ffmpeg est disponible.",
        "Fichiers transformés sans bloquer le reste de l'application.",
    ],
    [
        "Galerie",
        "Liste séparée des images et audios générés, avec pagination.",
        "Trace concrète du parcours complet: créer, retrouver, télécharger.",
    ],
    [
        "Équipe",
        "Profils, rôles, emails et photos chargées depuis static/Admins.",
        "Présentation propre du groupe dans l'application.",
    ],
]

TOOLS = [
    ["Flask / Jinja2", "Routes serveur, rendu HTML et formulaires."],
    ["Pillow", "Effets image et export des fichiers traités."],
    ["Pandas / NumPy", "Lecture et préparation des données numériques."],
    ["Matplotlib", "Création des visualisations à partir des données."],
    ["PyDub / ffmpeg", "Traitement audio lorsque la machine le permet."],
    ["unittest", "Tests des routes, formulaires, exports et nettoyages."],
    ["Git / GitHub", "Historique du travail et dépôt final à rendre."],
]

PIPELINE = [
    "L'utilisateur part du tableau de bord et choisit un atelier.",
    "Flask reçoit le formulaire et vérifie les valeurs utiles.",
    "Le module Python concerné génère ou transforme le contenu.",
    "Le résultat est enregistré dans static/generated.",
    "La page affiche le fichier final et propose le téléchargement.",
    "Les tests rejouent les parcours importants pour vérifier que rien ne casse.",
]

CHALLENGES = [
    [
        "Un app.py devenu trop chargé",
        "Séparer la configuration, les routes, les formulaires, le stockage et les modules métier.",
    ],
    [
        "Une interface qui devait paraître terminée",
        "Reprendre les textes en français et choisir un style plus sobre, plus proche d'un outil de travail.",
    ],
    [
        "L'audio dépend de ffmpeg",
        "Garder le module audio optionnel pour que l'application fonctionne même si ffmpeg n'est pas installé.",
    ],
    [
        "Les fichiers générés peuvent vite s'accumuler",
        "Limiter les caches, nettoyer les imports temporaires et paginer la galerie.",
    ],
]

DELIVERABLES = [
    ["Base de code complète", "Dépôt avec app.py, studio, modules, templates, static, tests et requirements.txt."],
    ["Application Flask", "Lancement par python app.py; les pages principales répondent correctement."],
    ["README", "Installation, lancement, tests, structure et livrables finaux."],
    ["Rapport final", "PDF de 2-3 pages couvrant le concept, les modules, le pipeline et les challenges."],
]


def format_report_date() -> str:
    return REPORT_DATE.strftime("%d/%m/%Y")


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


def build_markdown() -> None:
    lines = [
        "# Rapport final - Studio génératif interactif",
        "",
        f"- Date de vérification : {format_report_date()}",
        f"- Période de travail : {PROJECT_START} au {format_report_date()}",
        f"- Dépôt GitHub : {REPO_URL}",
        "- Objet : application Flask pour un projet de créativité numérique.",
        "",
        "## 1. Concept et direction artistique",
        "",
        "Le projet fonctionne comme un petit studio numérique. À partir de paramètres simples, d'un fichier CSV ou d'une image, "
        "l'utilisateur peut produire un rendu visuel puis le retrouver dans une galerie. La partie créative vient du fait que "
        "les réglages changent réellement le résultat, sans demander à l'utilisateur de toucher au code.",
        "",
        "La direction artistique est volontairement calme: interface claire, cartes sobres, couleurs limitées et textes en "
        "français. Ce choix rend l'application plus facile à présenter et évite que le décor prenne le dessus sur le "
        "fonctionnement.",
        "",
        "## 2. Modules implémentés",
        "",
        "| Module | Fonction | Résultat |",
        "| --- | --- | --- |",
    ]
    for module, function, result in MODULES:
        lines.append(f"| {module} | {function} | {result} |")

    lines.extend(
        [
            "",
            "## 3. Outils utilisés et pipeline technique",
            "",
            "| Outil | Utilisation |",
            "| --- | --- |",
        ]
    )
    for tool, usage in TOOLS:
        lines.append(f"| {tool} | {usage} |")

    lines.append("")
    lines.append("Pipeline :")
    for step in PIPELINE:
        lines.append(f"- {step}")

    lines.extend(
        [
            "",
            "## 4. Challenges rencontrés et solutions",
            "",
            "| Challenge | Solution |",
            "| --- | --- |",
        ]
    )
    for challenge, solution in CHALLENGES:
        lines.append(f"| {challenge} | {solution} |")

    lines.extend(
        [
            "",
            "## 5. Équipe et livrables",
            "",
            "| Photo | Membre | Rôle |",
            "| --- | --- | --- |",
        ]
    )
    for member in TEAM:
        lines.append(f"| {markdown_photo(member)} | {member['name']} | {member['role']} |")

    lines.extend(["", "| Livrable | Vérification |", "| --- | --- |"])
    for deliverable, verification in DELIVERABLES:
        lines.append(f"| {deliverable} | {verification} |")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#202124"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#202124"),
            spaceBefore=7,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.2,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHead",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.4,
            textColor=colors.HexColor("#202124"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.3,
            textColor=colors.HexColor("#202124"),
        )
    )
    return styles


STYLES = make_styles()


def paragraph(text: str, style: str = "BodySmall") -> Paragraph:
    return Paragraph(escape(text), STYLES[style])


def table_cell(value, *, header: bool = False):
    if hasattr(value, "wrap"):
        return value
    return Paragraph(escape(str(value)), STYLES["TableHead" if header else "TableText"])


def styled_table(data, widths):
    rows = []
    for row_index, row in enumerate(data):
        rows.append([table_cell(value, header=row_index == 0) for value in row])

    output = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    output.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd7e3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return output


def bullet_list(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(paragraph(item, "TableText"), leftIndent=8) for item in items],
        bulletType="bullet",
        leftIndent=12,
        bulletFontSize=5,
    )


def member_photo(member: dict[str, str]):
    photo_path = team_photo_path(member)
    if not photo_path:
        return paragraph("-", "TableText")

    image = Image(str(photo_path))
    max_width = 1.15 * cm
    max_height = 1.45 * cm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def build_pdf() -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.25 * cm,
        leftMargin=1.25 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.1 * cm,
        title="Rapport final - Studio génératif interactif",
    )
    story = []

    story.append(Paragraph("Rapport final", STYLES["ReportTitle"]))
    story.append(paragraph("Studio génératif interactif - projet Flask de créativité numérique"))
    story.append(
        styled_table(
            [
                ["Élément", "Information"],
                ["Date de vérification", format_report_date()],
                ["Période de travail", f"{PROJECT_START} au {format_report_date()}"],
                ["Dépôt GitHub", REPO_URL],
                ["Livrable", "Code complet, application Flask, README et rapport final de 2-3 pages."],
            ],
            [4.0 * cm, 13.6 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))

    story.append(Paragraph("1. Concept et direction artistique", STYLES["Section"]))
    story.append(
        paragraph(
            "Le projet fonctionne comme un petit studio numérique. L'utilisateur peut partir de réglages simples, "
            "d'un CSV ou d'une image, produire un rendu, puis le retrouver dans une galerie."
        )
    )
    story.append(Spacer(1, 0.08 * cm))
    story.append(
        paragraph(
            "La direction artistique reste calme: interface claire, cartes sobres, couleurs limitées et textes français. "
            "Le but est de montrer le fonctionnement du studio sans noyer la démonstration dans la décoration."
        )
    )

    story.append(Paragraph("2. Modules implémentés", STYLES["Section"]))
    story.append(styled_table([["Module", "Fonction", "Résultat"], *MODULES], [3.2 * cm, 8.1 * cm, 6.3 * cm]))

    story.append(PageBreak())
    story.append(Paragraph("3. Outils utilisés et pipeline technique", STYLES["Section"]))
    story.append(styled_table([["Outil", "Utilisation"], *TOOLS], [4.2 * cm, 13.4 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(paragraph("Pipeline de fonctionnement:"))
    story.append(bullet_list(PIPELINE))

    story.append(PageBreak())
    story.append(Paragraph("4. Challenges rencontrés et solutions", STYLES["Section"]))
    story.append(styled_table([["Challenge", "Solution"], *CHALLENGES], [5.7 * cm, 11.9 * cm]))

    story.append(Paragraph("5. Équipe et livrables vérifiés", STYLES["Section"]))
    story.append(
        styled_table(
            [["Photo", "Membre", "Rôle"]]
            + [[member_photo(member), member["name"], member["role"]] for member in TEAM],
            [1.55 * cm, 4.0 * cm, 12.05 * cm],
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(styled_table([["Livrable", "Vérification"], *DELIVERABLES], [4.5 * cm, 13.1 * cm]))

    doc.build(story)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_markdown()
    build_pdf()
    print(PDF_PATH)
    print(MD_PATH)

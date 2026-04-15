import math
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, t

st.set_page_config(page_title="Test sur une moyenne", layout="wide")


def format_percent_clean(alpha: float) -> str:
    val = 100 * alpha
    if abs(val - round(val)) < 1e-10:
        return f"{int(round(val))}%"
    txt = f"{val:.2f}".rstrip("0").rstrip(".")
    return f"{txt}%"


def format_prob_clean(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def format_number_clean(x: float, decimals: int = 4) -> str:
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def parse_observations(text: str) -> List[float]:
    """
    Convertit une chaîne de caractères en liste de nombres.
    Séparateurs acceptés : virgule, point-virgule, espace, retour ligne.
    """
    if not text.strip():
        return []

    cleaned = text.replace(";", " ").replace(",", " ").replace("\n", " ")
    parts = [p for p in cleaned.split() if p.strip()]

    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError("Les observations doivent être numériques.")


def test_moyenne_general(
    mu0: float,
    alpha: float = 0.05,
    alternative: str = "bilateral",
    observations: Optional[List[float]] = None,
    n: Optional[int] = None,
    moyenne_echantillon: Optional[float] = None,
    ecart_type: Optional[float] = None,
    variance: Optional[float] = None,
    sigma_connu: bool = False,
):
    """
    Test général sur une moyenne.

    Cas couverts :
    - observations fournies + sigma inconnu  -> Student
    - observations fournies + sigma connu    -> Normale, sigma à fournir
    - statistiques résumées + sigma connu    -> Normale
    - statistiques résumées + sigma inconnu  -> Student
    """

    if not (0 < alpha < 1):
        raise ValueError("Le niveau alpha doit être strictement compris entre 0 et 1.")

    if alternative not in {"bilateral", "left", "right"}:
        raise ValueError("Le type d'alternative est invalide.")

    variance_echantillon = None
    ecart_type_echantillon = None
    source = ""

    if observations is not None and len(observations) > 0:
        if len(observations) < 2:
            raise ValueError("Il faut au moins 2 observations.")

        n = len(observations)
        moyenne_echantillon = sum(observations) / n
        variance_echantillon = sum((x - moyenne_echantillon) ** 2 for x in observations) / (n - 1)
        ecart_type_echantillon = math.sqrt(variance_echantillon)
        source = "observations"

        if sigma_connu:
            if ecart_type is None and variance is None:
                raise ValueError(
                    "Si l'écart-type de la population est connu, il faut fournir l'écart-type ou la variance de la population."
                )
        else:
            # Si sigma est inconnu et qu'on a les observations,
            # on impose automatiquement l'écart-type empirique.
            ecart_type = ecart_type_echantillon
            variance = variance_echantillon

    else:
        source = "statistiques résumées"

        if n is None or moyenne_echantillon is None:
            raise ValueError(
                "Si les observations ne sont pas fournies, il faut renseigner n et la moyenne de l'échantillon."
            )

        if n < 2:
            raise ValueError("n doit être au moins égal à 2.")

        if ecart_type is None and variance is None:
            raise ValueError("Il faut fournir l'écart-type ou la variance.")

    if ecart_type is None and variance is not None:
        ecart_type = math.sqrt(variance)

    if ecart_type is None:
        raise ValueError("Impossible de déterminer l'écart-type.")

    if not sigma_connu and variance_echantillon is None:
        variance_echantillon = variance if variance is not None else ecart_type ** 2
        ecart_type_echantillon = ecart_type

    erreur_standard = ecart_type / math.sqrt(n)
    statistique = (moyenne_echantillon - mu0) / erreur_standard
    alpha_label = format_percent_clean(alpha)

    mu0_label = format_number_clean(mu0)

    if sigma_connu:
        loi = "Normale"
        ddl = None
        nom_stat = "Z"

        if alternative == "bilateral":
            quantile = norm.ppf(1 - alpha / 2)
            rejet = abs(statistique) > quantile
            p_value = 2 * (1 - norm.cdf(abs(statistique)))
            H1 = f"\\mu \\neq {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $|Z_{{obs}}| = {abs(statistique):.4f} > {quantile:.4f} = z_{{\alpha/2}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) différente de {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $|Z_{{obs}}| = {abs(statistique):.4f} < {quantile:.4f} = z_{{\alpha/2}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) différente de {mu0_label}."
                )

        elif alternative == "left":
            quantile = norm.ppf(alpha)
            rejet = statistique < quantile
            p_value = norm.cdf(statistique)
            H1 = f"\\mu < {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $Z_{{obs}} = {statistique:.4f} < {quantile:.4f} = -z_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) inférieure à {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $Z_{{obs}} = {statistique:.4f} > {quantile:.4f} = -z_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) inférieure à {mu0_label}."
                )

        else:
            quantile = norm.ppf(1 - alpha)
            rejet = statistique > quantile
            p_value = 1 - norm.cdf(statistique)
            H1 = f"\\mu > {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $Z_{{obs}} = {statistique:.4f} > {quantile:.4f} = z_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) supérieure à {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $Z_{{obs}} = {statistique:.4f} < {quantile:.4f} = z_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) supérieure à {mu0_label}."
                )

    else:
        loi = "Student"
        ddl = n - 1
        nom_stat = "T"

        if alternative == "bilateral":
            quantile = t.ppf(1 - alpha / 2, ddl)
            rejet = abs(statistique) > quantile
            p_value = 2 * (1 - t.cdf(abs(statistique), ddl))
            H1 = f"\\mu \\neq {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $|T_{{obs}}| = {abs(statistique):.4f} > {quantile:.4f} = t_{{\alpha/2}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) différente de {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $|T_{{obs}}| = {abs(statistique):.4f} < {quantile:.4f} = t_{{\alpha/2}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) différente de {mu0_label}."
                )

        elif alternative == "left":
            quantile = t.ppf(alpha, ddl)
            rejet = statistique < quantile
            p_value = t.cdf(statistique, ddl)
            H1 = f"\\mu < {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $T_{{obs}} = {statistique:.4f} < {quantile:.4f} = -t_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) inférieure à {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $T_{{obs}} = {statistique:.4f} > {quantile:.4f} = -t_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) inférieure à {mu0_label}."
                )

        else:
            quantile = t.ppf(1 - alpha, ddl)
            rejet = statistique > quantile
            p_value = 1 - t.cdf(statistique, ddl)
            H1 = f"\\mu > {mu0_label}"

            if rejet:
                conclusion = (
                    rf"Comme $T_{{obs}} = {statistique:.4f} > {quantile:.4f} = t_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de rejet. "
                    rf"On rejette donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne est statistiquement (ou significativement) supérieure à {mu0_label}."
                )
            else:
                conclusion = (
                    rf"Comme $T_{{obs}} = {statistique:.4f} < {quantile:.4f} = t_{{\alpha}}$, "
                    rf"la statistique observée appartient à la zone de non rejet. "
                    rf"On ne rejette pas donc $H_0$ au seuil de {alpha_label}. "
                    rf"La moyenne n'est pas statistiquement (ou significativement) supérieure à {mu0_label}."
                )

    return {
        "source": source,
        "n": n,
        "moyenne_echantillon": moyenne_echantillon,
        "variance_echantillon": variance_echantillon,
        "ecart_type_echantillon": ecart_type_echantillon,
        "ecart_type_utilise": ecart_type,
        "variance_utilisee": ecart_type ** 2,
        "erreur_standard": erreur_standard,
        "mu0": mu0,
        "mu0_label": mu0_label,
        "alpha": alpha,
        "alpha_label": alpha_label,
        "alternative": alternative,
        "sigma_connu": sigma_connu,
        "loi": loi,
        "ddl": ddl,
        "nom_statistique": nom_stat,
        "statistique_test": statistique,
        "quantile_critique": quantile,
        "p_value": p_value,
        "rejet_H0": rejet,
        "H1": H1,
        "conclusion": conclusion,
    }


def calcul_intervalle_confiance(resultats: dict):
    xbar = resultats["moyenne_echantillon"]
    alpha = resultats["alpha"]
    se = resultats["erreur_standard"]

    if resultats["sigma_connu"]:
        quantile_ic = norm.ppf(1 - alpha / 2)
        borne_inf = xbar - quantile_ic * se
        borne_sup = xbar + quantile_ic * se
        return {
            "quantile_ic": quantile_ic,
            "borne_inf": borne_inf,
            "borne_sup": borne_sup,
            "type": "normale",
        }
    else:
        ddl = resultats["ddl"]
        quantile_ic = t.ppf(1 - alpha / 2, ddl)
        borne_inf = xbar - quantile_ic * se
        borne_sup = xbar + quantile_ic * se
        return {
            "quantile_ic": quantile_ic,
            "borne_inf": borne_inf,
            "borne_sup": borne_sup,
            "type": "student",
        }


def tracer_distribution(resultats: dict):
    """
    Produit le graphique de la loi normale ou de Student
    avec statistique observée et zones de rejet.
    """
    statistique = resultats["statistique_test"]
    quantile = resultats["quantile_critique"]
    alpha = resultats["alpha"]
    alternative = resultats["alternative"]
    sigma_connu = resultats["sigma_connu"]
    ddl = resultats["ddl"]
    nom_stat = resultats["nom_statistique"]

    borne = max(4, abs(statistique) + 1, abs(quantile) + 1)
    x = np.linspace(-borne, borne, 3000)

    if sigma_connu:
        y = norm.pdf(x)
    else:
        y = t.pdf(x, ddl)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, y, color="black", linewidth=1.5)

    alpha_pct = format_percent_clean(alpha)
    alpha_half_pct = format_percent_clean(alpha / 2)
    non_rejet_pct = format_percent_clean(1 - alpha)

    if alternative == "bilateral":
        mask_L = x <= -quantile
        mask_C = (x >= -quantile) & (x <= quantile)
        mask_R = x >= quantile

        ax.fill_between(
            x, 0, y, where=mask_L, color="gray", alpha=0.5,
            label=f"Zone de rejet (α/2 = {alpha_half_pct})"
        )
        ax.fill_between(x, 0, y, where=mask_R, color="gray", alpha=0.5)
        ax.fill_between(
            x, 0, y, where=mask_C, color="lightblue", alpha=0.3,
            label=f"Zone de non-rejet ({non_rejet_pct})"
        )

        ax.axvline(-quantile, color="black", linewidth=1.2, linestyle="--")
        ax.axvline(quantile, color="black", linewidth=1.2, linestyle="--")

        ax.text(-quantile - 0.5, max(y) * 0.12, alpha_half_pct, fontsize=9, ha="center", color="dimgray")
        ax.text(quantile + 0.5, max(y) * 0.12, alpha_half_pct, fontsize=9, ha="center", color="dimgray")

        ax.text(-borne * 0.72, max(y) * 0.25, "Rejet\nH₀", fontsize=10, ha="center", va="center", color="dimgray")
        ax.text(borne * 0.72, max(y) * 0.25, "Rejet\nH₀", fontsize=10, ha="center", va="center", color="dimgray")
        ax.text(
            0, max(y) * 0.55, f"Non-rejet de H₀\n({non_rejet_pct})",
            fontsize=10, ha="center", va="center", color="steelblue", fontweight="bold"
        )

        ax.set_xticks([-quantile, 0, quantile])
        ax.set_xticklabels([f"{-quantile:.2f}", "0", f"{quantile:.2f}"])

    elif alternative == "left":
        mask_L = x <= quantile
        mask_C = x >= quantile

        ax.fill_between(
            x, 0, y, where=mask_L, color="gray", alpha=0.5,
            label=f"Zone de rejet (α = {alpha_pct})"
        )
        ax.fill_between(
            x, 0, y, where=mask_C, color="lightblue", alpha=0.3,
            label=f"Zone de non-rejet ({non_rejet_pct})"
        )

        ax.axvline(quantile, color="black", linewidth=1.2, linestyle="--")
        ax.text(quantile - 0.5, max(y) * 0.12, alpha_pct, fontsize=9, ha="center", color="dimgray")

        ax.text(-borne * 0.72, max(y) * 0.25, "Rejet\nH₀", fontsize=10, ha="center", va="center", color="dimgray")
        ax.text(
            borne * 0.25, max(y) * 0.55, f"Non-rejet de H₀\n({non_rejet_pct})",
            fontsize=10, ha="center", va="center", color="steelblue", fontweight="bold"
        )

        ax.set_xticks([quantile, 0])
        ax.set_xticklabels([f"{quantile:.2f}", "0"])

    else:
        mask_C = x <= quantile
        mask_R = x >= quantile

        ax.fill_between(
            x, 0, y, where=mask_R, color="gray", alpha=0.5,
            label=f"Zone de rejet (α = {alpha_pct})"
        )
        ax.fill_between(
            x, 0, y, where=mask_C, color="lightblue", alpha=0.3,
            label=f"Zone de non-rejet ({non_rejet_pct})"
        )

        ax.axvline(quantile, color="black", linewidth=1.2, linestyle="--")
        ax.text(quantile + 0.5, max(y) * 0.12, alpha_pct, fontsize=9, ha="center", color="dimgray")

        ax.text(borne * 0.72, max(y) * 0.25, "Rejet\nH₀", fontsize=10, ha="center", va="center", color="dimgray")
        ax.text(
            -borne * 0.25, max(y) * 0.55, f"Non-rejet de H₀\n({non_rejet_pct})",
            fontsize=10, ha="center", va="center", color="steelblue", fontweight="bold"
        )

        ax.set_xticks([0, quantile])
        ax.set_xticklabels(["0", f"{quantile:.2f}"])

    ax.axvline(
        statistique, color="red", linewidth=1.8, linestyle="-",
        label=f"{nom_stat}_obs = {statistique:.4f}"
    )

    decalage = 0.12 if statistique <= 0 else -0.12
    align = "left" if statistique <= 0 else "right"

    ax.text(
        statistique + decalage,
        max(y) * 0.45,
        f"{nom_stat}_obs = {statistique:.4f}",
        fontsize=9,
        ha=align,
        va="center",
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="red",
            linewidth=0.7,
            alpha=0.9,
        ),
    )

    mu0_label = resultats["mu0_label"]
    loi_texte = "Loi normale centrée réduite" if sigma_connu else f"Loi de Student à {ddl} ddl"

    if alternative == "bilateral":
        titre_test = f"Test bilatéral : H₀ : μ = {mu0_label} contre H₁ : μ ≠ {mu0_label}"
    elif alternative == "left":
        titre_test = f"Test unilatéral gauche : H₀ : μ = {mu0_label} contre H₁ : μ < {mu0_label}"
    else:
        titre_test = f"Test unilatéral droit : H₀ : μ = {mu0_label} contre H₁ : μ > {mu0_label}"

    ax.set_title(
        f"{titre_test}\n{loi_texte} (α = {alpha_pct}, {nom_stat.lower()}_obs = {statistique:.4f})",
        fontsize=11
    )
    ax.set_xlabel("z" if sigma_connu else "t")
    ax.set_ylabel("Densité")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# -------------------------
# Interface Streamlit
# -------------------------

st.title("Application Streamlit : test sur une moyenne")

st.markdown(
    """
Cette application permet d'effectuer un test sur une moyenne.

Deux possibilités d'entrée :

- **Observations**
- **Statistiques résumées**
"""
)

col1, col2 = st.columns(2)

with col1:
    mode_entree = st.selectbox(
        "Choix de l'entrée",
        ["Observations", "Statistiques résumées"]
    )

    mu0 = st.number_input("Valeur sous H₀ : μ₀", value=46.0, step=0.1)

    alpha = st.number_input(
        "Niveau du test α",
        min_value=0.001,
        max_value=0.50,
        value=0.05,
        step=0.01
    )

    alternative_label = st.selectbox(
        "Type de test",
        [
            "Bilatéral : H₁ : μ ≠ μ₀",
            "Unilatéral gauche : H₁ : μ < μ₀",
            "Unilatéral droit : H₁ : μ > μ₀",
        ],
    )

    alternative_map = {
        "Bilatéral : H₁ : μ ≠ μ₀": "bilateral",
        "Unilatéral gauche : H₁ : μ < μ₀": "left",
        "Unilatéral droit : H₁ : μ > μ₀": "right",
    }
    alternative = alternative_map[alternative_label]

    sigma_connu_label = st.selectbox(
        "Statut de l'écart-type de la population",
        ["Écart-type connu", "Écart-type inconnu"]
    )
    sigma_connu = sigma_connu_label == "Écart-type connu"

with col2:
    observations = None
    n = None
    moyenne_echantillon = None
    ecart_type = None
    variance = None
    texte_obs = ""

    if mode_entree == "Observations":
        texte_obs = st.text_area(
            "Observations",
            value="43 46 48 40 50 52 43 46 44 49 50 54 45 48 53 44 48 48",
            height=140,
            help="Séparer les valeurs par des espaces, virgules, points-virgules ou retours à la ligne. Si vous laissez cette case vide, vous pouvez renseigner directement n et la moyenne empirique."
        )

        if texte_obs.strip() == "":
            n = st.number_input(
                "Taille de l'échantillon n",
                min_value=2,
                value=18,
                step=1
            )

            moyenne_echantillon = st.number_input(
                "Moyenne empirique de l'échantillon",
                value=47.2778,
                step=0.1
            )

            if sigma_connu:
                type_dispersion = st.selectbox(
                    "Information complémentaire",
                    ["Écart-type de la population", "Variance de la population"]
                )
            else:
                type_dispersion = st.selectbox(
                    "Information complémentaire",
                    ["Écart-type empirique de l'échantillon", "Variance empirique de l'échantillon"]
                )

            if type_dispersion == "Écart-type de la population":
                ecart_type = st.number_input(
                    "Valeur de l'écart-type de la population",
                    min_value=0.0,
                    value=1.0,
                    step=0.1
                )
            elif type_dispersion == "Variance de la population":
                variance = st.number_input(
                    "Valeur de la variance de la population",
                    min_value=0.0,
                    value=1.0,
                    step=0.1
                )
            elif type_dispersion == "Écart-type empirique de l'échantillon":
                ecart_type = st.number_input(
                    "Valeur de l'écart-type empirique de l'échantillon",
                    min_value=0.0,
                    value=4.0,
                    step=0.1
                )
            elif type_dispersion == "Variance empirique de l'échantillon":
                variance = st.number_input(
                    "Valeur de la variance empirique de l'échantillon",
                    min_value=0.0,
                    value=16.0,
                    step=0.1
                )

        else:
            if sigma_connu:
                choix_optionnel = st.selectbox(
                    "Information complémentaire",
                    ["Écart-type de la population", "Variance de la population"]
                )

                if choix_optionnel == "Écart-type de la population":
                    ecart_type = st.number_input(
                        "Valeur de l'écart-type de la population",
                        min_value=0.0,
                        value=1.0,
                        step=0.1
                    )
                elif choix_optionnel == "Variance de la population":
                    variance = st.number_input(
                        "Valeur de la variance de la population",
                        min_value=0.0,
                        value=1.0,
                        step=0.1
                    )
            else:
                st.info(
                    "Comme les observations sont fournies et que l'écart-type de la population est inconnu, "
                    "l'application utilise automatiquement l'écart-type empirique calculé sur l'échantillon."
                )

    else:
        n = st.number_input(
            "Taille de l'échantillon n",
            min_value=2,
            value=18,
            step=1
        )

        moyenne_echantillon = st.number_input(
            "Moyenne empirique de l'échantillon",
            value=47.2778,
            step=0.1
        )

        if sigma_connu:
            type_dispersion = st.selectbox(
                "Information complémentaire",
                ["Écart-type de la population", "Variance de la population"]
            )
        else:
            type_dispersion = st.selectbox(
                "Information complémentaire",
                ["Écart-type empirique de l'échantillon", "Variance empirique de l'échantillon"]
            )

        if type_dispersion == "Écart-type de la population":
            ecart_type = st.number_input(
                "Valeur de l'écart-type de la population",
                min_value=0.0,
                value=1.0,
                step=0.1
            )
        elif type_dispersion == "Variance de la population":
            variance = st.number_input(
                "Valeur de la variance de la population",
                min_value=0.0,
                value=1.0,
                step=0.1
            )
        elif type_dispersion == "Écart-type empirique de l'échantillon":
            ecart_type = st.number_input(
                "Valeur de l'écart-type empirique de l'échantillon",
                min_value=0.0,
                value=4.0,
                step=0.1
            )
        elif type_dispersion == "Variance empirique de l'échantillon":
            variance = st.number_input(
                "Valeur de la variance empirique de l'échantillon",
                min_value=0.0,
                value=16.0,
                step=0.1
            )

st.markdown("---")

if st.button("Effectuer le test"):
    try:
        if mode_entree == "Observations" and texte_obs.strip() != "":
            observations = parse_observations(texte_obs)
        else:
            observations = None

        resultats = test_moyenne_general(
            mu0=mu0,
            alpha=alpha,
            alternative=alternative,
            observations=observations,
            n=n,
            moyenne_echantillon=moyenne_echantillon,
            ecart_type=ecart_type,
            variance=variance,
            sigma_connu=sigma_connu,
        )

        ic = calcul_intervalle_confiance(resultats)

        st.subheader("Résultats numériques")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n", f"{resultats['n']}")
        c2.metric("Moyenne", f"{resultats['moyenne_echantillon']:.4f}")
        c3.metric(resultats["nom_statistique"], f"{resultats['statistique_test']:.4f}")
        c4.metric("p-value", f"{resultats['p_value']:.4f}")

        st.write(f"**Loi utilisée :** {resultats['loi']}")
        if resultats["ddl"] is not None:
            st.write(f"**Degrés de liberté :** {resultats['ddl']}")
        st.write(f"**Quantile critique :** {resultats['quantile_critique']:.4f}")

        if resultats["source"] == "observations":
            st.write(f"**Variance empirique de l'échantillon :** {resultats['variance_echantillon']:.4f}")
            st.write(f"**Écart-type empirique de l'échantillon :** {resultats['ecart_type_echantillon']:.4f}")
        else:
            if resultats["sigma_connu"]:
                st.write(f"**Variance de la population utilisée :** {resultats['variance_utilisee']:.4f}")
                st.write(f"**Écart-type de la population utilisé :** {resultats['ecart_type_utilise']:.4f}")
            else:
                st.write(f"**Variance empirique de l'échantillon utilisée :** {resultats['variance_utilisee']:.4f}")
                st.write(f"**Écart-type empirique de l'échantillon utilisé :** {resultats['ecart_type_utilise']:.4f}")

        # -------------------------------------------------------
        # INTERVALLE DE CONFIANCE AVANT LA MOYENNE OBSERVÉE
        # -------------------------------------------------------
        st.subheader("Intervalle de confiance pour la moyenne")

        niveau_label = format_percent_clean(1 - resultats["alpha"])
        mu0_label = resultats["mu0_label"]

        if resultats["sigma_connu"]:
            st.markdown(
                f"**Intervalle de confiance à niveau {niveau_label} :**"
            )

            st.latex(
                r"\frac{\bar X - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1)"
            )

            st.latex(
                r"""
\Rightarrow
P\left(
- z_{\frac{\alpha}{2}}
\le
\frac{\bar X - \mu}{\sigma/\sqrt{n}}
\le
z_{\frac{\alpha}{2}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
P\left(
- z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\le
\bar X - \mu
\le
z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
P\left(
\bar X - z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\le
\mu
\le
\bar X + z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
IC_{1-\alpha}(\mu)
=
\left[
\bar{X} - z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\ ;\
\bar{X} + z_{\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}
\right]
"""
            )

            st.markdown(f"**Intervalle de confiance à niveau {niveau_label} :**")

            st.latex(
                rf"""
IC_{{{niveau_label}}}(\mu)
=
\left[
\bar{{X}} - z_{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{\sigma}}{{\sqrt{{n}}}}
\ ;\
\bar{{X}} + z_{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{\sigma}}{{\sqrt{{n}}}}
\right]
"""
            )

            st.markdown("**Après observation :**")

            st.latex(
                rf"""
IC_{{{niveau_label}}}(\mu)
=
\left[
\bar{{x}} - z_{{{1 - resultats['alpha']/2:.3f}}}\times\frac{{\sigma}}{{\sqrt{{n}}}}
\ ;\
\bar{{x}} + z_{{{1 - resultats['alpha']/2:.3f}}}\times\frac{{\sigma}}{{\sqrt{{n}}}}
\right]
"""
            )

            st.markdown(
                rf"En substituant $\bar{{x}} = {resultats['moyenne_echantillon']:.4f}$, "
                rf"$z_{{{format_prob_clean(resultats['alpha']/2)}}} = {ic['quantile_ic']:.4f}$, "
                rf"$\sigma = {resultats['ecart_type_utilise']:.4f}$ et $n = {resultats['n']}$ :"
            )

            st.latex(
                rf"""
\begin{{aligned}}
IC_{{{niveau_label}}}(\mu)
&= \left[
{resultats['moyenne_echantillon']:.4f}
- {ic['quantile_ic']:.4f}\times\frac{{{resultats['ecart_type_utilise']:.4f}}}{{\sqrt{{{resultats['n']}}}}}
\ ;\
{resultats['moyenne_echantillon']:.4f}
+ {ic['quantile_ic']:.4f}\times\frac{{{resultats['ecart_type_utilise']:.4f}}}{{\sqrt{{{resultats['n']}}}}}
\right] \\[6pt]
&= \left[
{resultats['moyenne_echantillon']:.4f}
- {ic['quantile_ic']:.4f}\times {resultats['erreur_standard']:.4f}
\ ;\
{resultats['moyenne_echantillon']:.4f}
+ {ic['quantile_ic']:.4f}\times {resultats['erreur_standard']:.4f}
\right] \\[6pt]
&= \left[
{resultats['moyenne_echantillon']:.4f - ic['quantile_ic'] * resultats['erreur_standard']:.4f}
\ ;\
{resultats['moyenne_echantillon']:.4f + ic['quantile_ic'] * resultats['erreur_standard']:.4f}
\right]
\end{{aligned}}
"""
            )

            st.latex(
                rf"\boxed{{IC_{{{niveau_label}}}(\mu)=\left[{ic['borne_inf']:.4f}\ ;\ {ic['borne_sup']:.4f}\right]}}"
            )

        else:
            st.markdown(
                f"**Intervalle de confiance à niveau {niveau_label} :**"
            )

            st.latex(
                rf"\frac{{\bar X - \mu}}{{S/\sqrt{{n}}}} \sim t_{{{resultats['ddl']}}}"
            )

            st.latex(
                r"""
\Rightarrow
P\left(
- t_{n-1}^{\frac{\alpha}{2}}
\le
\frac{\bar X - \mu}{S/\sqrt{n}}
\le
t_{n-1}^{\frac{\alpha}{2}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
P\left(
- t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\le
\bar X - \mu
\le
t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
P\left(
\bar X - t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\le
\mu
\le
\bar X + t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\right)
=
1-\alpha
"""
            )

            st.latex(
                r"""
\Rightarrow
IC_{1-\alpha}(\mu)
=
\left[
\bar{X} - t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\ ;\
\bar{X} + t_{n-1}^{\frac{\alpha}{2}} \frac{S}{\sqrt{n}}
\right]
"""
            )

            st.markdown(f"**Intervalle de confiance à niveau {niveau_label} :**")

            st.latex(
                rf"""
IC_{{{niveau_label}}}(\mu)
=
\left[
\bar{{X}} - t_{{n-1}}^{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{S}}{{\sqrt{{n}}}}
\ ;\
\bar{{X}} + t_{{n-1}}^{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{S}}{{\sqrt{{n}}}}
\right]
"""
            )

            st.markdown("**Après observation :**")

            st.latex(
                rf"""
IC_{{{niveau_label}}}(\mu)
=
\left[
\bar{{x}} - t_{{{resultats['ddl']}}}^{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{s}}{{\sqrt{{n}}}}
\ ;\
\bar{{x}} + t_{{{resultats['ddl']}}}^{{{format_prob_clean(resultats['alpha']/2)}}}\times\frac{{s}}{{\sqrt{{n}}}}
\right]
"""
            )

            st.markdown(
                rf"En substituant $\bar{{x}} = {resultats['moyenne_echantillon']:.4f}$, "
                rf"$t_{{{resultats['ddl']}}}^{{{format_prob_clean(resultats['alpha']/2)}}} \approx {ic['quantile_ic']:.4f}$, "
                rf"$s = {resultats['ecart_type_utilise']:.4f}$ et $n = {resultats['n']}$ :"
            )

            st.latex(
                rf"""
\begin{{aligned}}
IC_{{{niveau_label}}}(\mu)
&= \left[
{resultats['moyenne_echantillon']:.4f}
- {ic['quantile_ic']:.4f}\times\frac{{{resultats['ecart_type_utilise']:.4f}}}{{\sqrt{{{resultats['n']}}}}}
\ ;\
{resultats['moyenne_echantillon']:.4f}
+ {ic['quantile_ic']:.4f}\times\frac{{{resultats['ecart_type_utilise']:.4f}}}{{\sqrt{{{resultats['n']}}}}}
\right] \\[6pt]
&= \left[
{resultats['moyenne_echantillon']:.4f}
- {ic['quantile_ic']:.4f}\times {resultats['erreur_standard']:.4f}
\ ;\
{resultats['moyenne_echantillon']:.4f}
+ {ic['quantile_ic']:.4f}\times {resultats['erreur_standard']:.4f}
\right] \\[6pt]
&= \left[
{resultats['moyenne_echantillon']:.4f - ic['quantile_ic'] * resultats['erreur_standard']:.4f}
\ ;\
{resultats['moyenne_echantillon']:.4f + ic['quantile_ic'] * resultats['erreur_standard']:.4f}
\right]
\end{{aligned}}
"""
            )

            st.latex(
                rf"\boxed{{IC_{{{niveau_label}}}(\mu)=\left[{ic['borne_inf']:.4f}\ ;\ {ic['borne_sup']:.4f}\right]}}"
            )

        if resultats["alternative"] == "bilateral":
            st.markdown("**Test de $H_0 : \mu = \mu_0$ à l’aide de l’intervalle de confiance :**")
            st.markdown(
                rf"Sur la base de l’intervalle de confiance précédent, on rejette "
                rf"$H_0 : \mu = \mu_0$ au niveau $\alpha = {resultats['alpha_label']}$ "
                rf"si $\mu_0 \notin IC_{{{niveau_label}}}(\mu)$."
            )
            st.markdown("Il y a 2 cas :")
            st.markdown(
                rf"i) $\mu_0 \in IC_{{{niveau_label}}}(\mu)$ $\implies$ **on ne rejette pas** "
                rf"$H_0$ au niveau $\alpha = {resultats['alpha_label']}$."
            )
            st.markdown(
                rf"ii) $\mu_0 \notin IC_{{{niveau_label}}}(\mu)$ $\implies$ **on rejette** "
                rf"$H_0$ au niveau $\alpha = {resultats['alpha_label']}$."
            )

            if ic["borne_inf"] <= resultats["mu0"] <= ic["borne_sup"]:
                st.markdown("**Application à notre cas :**")
                st.latex(
                    rf"{ic['borne_inf']:.4f} \le {mu0_label} \le {ic['borne_sup']:.4f}"
                )
                st.markdown(
                    rf"Comme $\mu_0 = {mu0_label}$ appartient à l’intervalle de confiance "
                    rf"$IC_{{{niveau_label}}}(\mu)$, **on ne rejette pas** $H_0$ "
                    rf"au niveau $\alpha = {resultats['alpha_label']}$."
                )
            else:
                st.markdown("**Application à notre cas :**")
                st.latex(
                    rf"{mu0_label} \notin \left[{ic['borne_inf']:.4f}\ ;\ {ic['borne_sup']:.4f}\right]"
                )
                st.markdown(
                    rf"Comme $\mu_0 = {mu0_label}$ n’appartient pas à l’intervalle de confiance "
                    rf"$IC_{{{niveau_label}}}(\mu)$, **on rejette** $H_0$ "
                    rf"au niveau $\alpha = {resultats['alpha_label']}$."
                )

        # -------------------------------------------------------
        # PRÉSENTATION DÉTAILLÉE DU TEST
        # -------------------------------------------------------
        st.subheader("Présentation détaillée du test")

        st.markdown("**Moyenne observée dans l’échantillon :**")
        st.latex(rf"\bar{{x}} \approx {resultats['moyenne_echantillon']:.4f}")

        st.markdown(f"**Test sur la moyenne au niveau α = {resultats['alpha_label']}**")

        st.markdown("**i) Les hypothèses du test :**")
        st.latex(rf"H_0 :\ \mu = {resultats['mu0_label']}")
        st.write("contre")
        st.latex(rf"H_1 :\ {resultats['H1']}")

        st.markdown("**ii) Statistique de test : loi et calcul**")

        if resultats["sigma_connu"]:
            st.write("Sous H₀, la statistique de test suit une loi normale centrée réduite :")
            st.latex(
                rf"Z = \frac{{\bar{{X}} - \mu_0}}{{\sigma/\sqrt{{n}}}} \sim_{{H_0}} \mathcal{{N}}(0,1), \qquad \mu_0 = {resultats['mu0_label']}"
            )
            st.write("Calculons la statistique de test :")
            st.latex(
                rf"Z_{{obs}} = \frac{{{resultats['moyenne_echantillon']:.4f} - {resultats['mu0_label']}}}{{{resultats['erreur_standard']:.5f}}}"
                rf"\approx {resultats['statistique_test']:.4f}"
            )
        else:
            st.write("Sous H₀, la statistique de test suit une loi de Student :")
            st.latex(
                rf"T = \frac{{\bar{{X}} - \mu_0}}{{S/\sqrt{{n}}}} \sim_{{H_0}} t_{{{resultats['ddl']}}}, \qquad \mu_0 = {resultats['mu0_label']}"
            )
            st.write("Calculons la statistique de test :")
            st.latex(
                rf"T_{{obs}} = \frac{{{resultats['moyenne_echantillon']:.4f} - {resultats['mu0_label']}}}{{{resultats['erreur_standard']:.5f}}}"
                rf"\approx {resultats['statistique_test']:.4f}"
            )

        st.markdown("**iii) Règle de décision**")

        stat_sym = "Z" if resultats["sigma_connu"] else "T"
        quant_sym = "z" if resultats["sigma_connu"] else "t"

        if resultats["alternative"] == "bilateral":
            st.markdown(
                rf"Comme le test est bilatéral, on prend le quantile ${quant_sym}_{{\alpha/2}}$."
            )
            st.markdown(
                rf"On rejette $H_0$ si $|{stat_sym}_{{obs}}| > {quant_sym}_{{\alpha/2}} = {quant_sym}_{{{format_prob_clean(resultats['alpha']/2)}}} = {resultats['quantile_critique']:.4f}$."
            )
            symbole = ">" if resultats["rejet_H0"] else "<"
            st.latex(
                rf"|{stat_sym}_{{obs}}| = {abs(resultats['statistique_test']):.4f} {symbole} {resultats['quantile_critique']:.4f}"
            )

        elif resultats["alternative"] == "left":
            st.markdown(
                rf"Comme le test est unilatéral gauche, on prend le quantile $-{quant_sym}_{{\alpha}}$."
            )
            st.markdown(
                rf"On rejette $H_0$ si ${stat_sym}_{{obs}} < -{quant_sym}_{{\alpha}} = -{quant_sym}_{{{format_prob_clean(resultats['alpha'])}}} = {resultats['quantile_critique']:.4f}$."
            )
            symbole = "<" if resultats["rejet_H0"] else ">"
            st.latex(
                rf"{stat_sym}_{{obs}} = {resultats['statistique_test']:.4f} {symbole} {resultats['quantile_critique']:.4f}"
            )

        else:
            st.markdown(
                rf"Comme le test est unilatéral droit, on prend le quantile ${quant_sym}_{{\alpha}}$."
            )
            st.markdown(
                rf"On rejette $H_0$ si ${stat_sym}_{{obs}} > {quant_sym}_{{\alpha}} = {quant_sym}_{{{format_prob_clean(resultats['alpha'])}}} = {resultats['quantile_critique']:.4f}$."
            )
            symbole = ">" if resultats["rejet_H0"] else "<"
            st.latex(
                rf"{stat_sym}_{{obs}} = {resultats['statistique_test']:.4f} {symbole} {resultats['quantile_critique']:.4f}"
            )

        st.markdown("**Conclusion :**")
        st.markdown(resultats["conclusion"])

        st.subheader("Graphique de la distribution du test")
        fig = tracer_distribution(resultats)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur : {e}")

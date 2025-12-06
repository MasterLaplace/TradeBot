#!/bin/bash

# Script qui lance un nouveau shell avec le venv activé
# Utilisation: ./activate_venv.sh

VENV_DIR="venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/$VENV_DIR"

REQUIREMENT_FILE="$SCRIPT_DIR/requirement.txt"

# Vérifier si le venv existe déjà
if [ ! -d "$VENV_PATH" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_PATH"

    if [ $? -ne 0 ]; then
        echo "✗ Erreur lors de la création de l'environnement virtuel"
        exit 1
    fi
    echo "✓ Environnement virtuel créé avec succès"
else
    # Si le venv existe mais ne contient pas les scripts d'activation, re-créer le venv
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "Le venv existe mais le script d'activation est manquant — recréation de l'environnement virtuel..."
        rm -rf "$VENV_PATH"
        python3 -m venv "$VENV_PATH"
        if [ $? -ne 0 ]; then
            echo "✗ Erreur lors de la création de l'environnement virtuel"
            exit 1
        fi
        echo "✓ Environnement virtuel recréé avec succès"
    fi
fi

# Mettre à jour pip vers la dernière version (utilise python -m pip pour plus de robustesse)
echo "Mise à jour de pip vers la dernière version..."
if [ -x "$VENV_PATH/bin/python" ]; then
    # si pip est absent, utiliser ensurepip pour l'installer
    if [ ! -x "$VENV_PATH/bin/pip" ] && [ ! -x "$VENV_PATH/bin/pip3" ]; then
        echo "pip introuvable dans le venv — tentative d'installation avec ensurepip..."
        "$VENV_PATH/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi
    # préférer la commande python -m pip (compatible) :
    "$VENV_PATH/bin/python" -m pip install --upgrade pip --quiet
    if [ $? -eq 0 ]; then
        echo "✓ pip mis à jour avec succès"
    else
        echo "⚠ Avertissement: Impossible de mettre à jour pip (continuation avec la version actuelle)"
    fi
else
    echo "✗ Erreur: impossible de trouver 'python' dans le venv ($VENV_PATH/bin/python introuvable)."
    exit 1
fi

# Installer ou mettre à jour les dépendances dans le venv
if [ -f "$REQUIREMENT_FILE" ]; then
    echo "Installation des dépendances dans le venv..."
    if [ -x "$VENV_PATH/bin/python" ]; then
        "$VENV_PATH/bin/python" -m pip install -r "$REQUIREMENT_FILE"
        if [ $? -eq 0 ]; then
            echo "✓ Dépendances installées avec succès"
        else
            echo "✗ Erreur lors de l'installation des dépendances"
            exit 1
        fi
    else
        echo "✗ Erreur: impossible de trouver 'python' dans le venv, impossible d'installer les dépendances"
        exit 1
    fi
fi

# Lancer un nouveau shell avec le venv activé
echo "Lancement d'un nouveau shell avec le venv activé..."
echo "Tapez 'exit' pour quitter le shell et revenir à votre shell précédent."
echo ""

exec $SHELL -c "source '$VENV_PATH/bin/activate' && cd '$SCRIPT_DIR' && exec $SHELL"

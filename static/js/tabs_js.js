/**
 * Gestionnaire pour les onglets de l'interface
 * Permet de naviguer entre les différents modes de prédiction
 * et gère également le comportement du sélecteur de fichier
 */
document.addEventListener('DOMContentLoaded', function() {
    // Gestion des onglets
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.getAttribute('data-tab');
            
            // Désactiver tous les onglets
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Activer l'onglet sélectionné
            this.classList.add('active');
            document.getElementById(tab).classList.add('active');
        });
    });

    // Gestion de l'affichage du nom de fichier
    const fileInput = document.getElementById('file');
    const fileText = document.querySelector('.file-input-text');

    if (fileInput && fileText) {
        fileInput.addEventListener('change', function() {
            fileText.textContent = this.files.length > 0 ? this.files[0].name : 'Sélectionner un fichier';
        });
    }
});
/**
 * Gestionnaire de changement de thème
 * Permet de basculer entre les modes clair et sombre
 * et mémorise le choix de l'utilisateur
 */
document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.querySelector('.theme-toggle');
    
    // Appliquer le thème sauvegardé
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        toggleButton.textContent = '☀️';
    }
    
    // Changement de thème au clic
    toggleButton.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        toggleButton.textContent = isDark ? '☀️' : '🌙';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
    
    // Détecter la préférence système
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Si aucun thème n'est sauvegardé, utiliser la préférence système
    if (!savedTheme) {
        if (prefersDarkScheme.matches) {
            document.body.classList.add('dark-mode');
            toggleButton.textContent = '☀️';
            localStorage.setItem('theme', 'dark');
        }
    }
    
    // Écouter les changements de préférence système
    prefersDarkScheme.addEventListener('change', (e) => {
        // Ne changer le thème que si l'utilisateur n'a pas fait de choix explicite
        if (!localStorage.getItem('theme')) {
            if (e.matches) {
                document.body.classList.add('dark-mode');
                toggleButton.textContent = '☀️';
            } else {
                document.body.classList.remove('dark-mode');
                toggleButton.textContent = '🌙';
            }
        }
    });
});
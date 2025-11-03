(function(){
	const root = document.documentElement;
	const key = 'weatherdash-theme';
	const toggle = document.getElementById('themeToggle');
	const saved = localStorage.getItem(key);
	if(saved){
		root.setAttribute('data-bs-theme', saved);
		if(toggle) toggle.checked = saved === 'dark';
	}else{
		const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
		root.setAttribute('data-bs-theme', prefersDark ? 'dark' : 'light');
		if(toggle) toggle.checked = prefersDark;
	}
	if(toggle){
		toggle.addEventListener('change', function(){
			const theme = toggle.checked ? 'dark' : 'light';
			root.setAttribute('data-bs-theme', theme);
			localStorage.setItem(key, theme);
		});
	}
})();


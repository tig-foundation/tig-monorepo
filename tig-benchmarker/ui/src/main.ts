import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent, appConfig).catch((err) =>
  console.error(err)
);

(window as any).MonacoEnvironment = {
  getWorkerUrl: function (_moduleId: string, label: string) {
    if (label === 'json') {
      return '/assets/monaco/vs/language/json/json.worker.js';
    }
    if (label === 'css') {
      return '/assets/monaco/vs/language/css/css.worker.js';
    }
    if (label === 'html') {
      return '/assets/monaco/vs/language/html/html.worker.js';
    }
    if (label === 'typescript' || label === 'javascript') {
      return '/assets/monaco/vs/language/typescript/ts.worker.js';
    }
    return '/assets/monaco/vs/editor/editor.worker.js';
  },
};

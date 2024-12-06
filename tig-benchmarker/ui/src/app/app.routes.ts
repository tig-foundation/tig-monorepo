import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { ConfigComponent } from './pages/config/config.component';

export const routes: Routes = [
  {
    path: 'home',
    component: HomeComponent,
  },
  {
    path: 'config',
    component: ConfigComponent,
  },
  { path: '**', redirectTo: 'home' },
];

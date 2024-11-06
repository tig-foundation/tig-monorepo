import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { SettingsComponent } from './pages/settings/settings.component';
import { SlaveManagerComponent } from './pages/slave-manager/slave-manager.component';
import { JobManagerComponent } from './pages/job-manager/job-manager.component';
import { AuthComponent } from './components/auth/auth.component';

export const routes: Routes = [
  {
    path: 'auth',
    component: AuthComponent,
  },
  {
    path: 'home',
    component: HomeComponent,
  },
  {
    path: 'jobs',
    component: JobManagerComponent,
  },
  {
    path: 'slaves',
    component: SlaveManagerComponent,
  },
  {
    path: 'settings',
    component: SettingsComponent,
  },
  { path: '**', redirectTo: 'home' },
];
